import time
import numpy as np
import sys
import pickle
from tensorflow.keras.layers import LSTM, Bidirectional
from random import shuffle
import time
# add encode path
sys.path.insert(0, './Encode/')
import attention_layer
import ffn_layer
import embedding_layer
import model_utils
import transformer_encode
import hparam
import metric
import tensorflow as tf
from tensorflow.keras.layers import Conv1D
flags = tf.compat.v1.flags
logging = tf.compat.v1.logging

#Process sequence length
def cut(a, maxlen):
   b=[]
   for i in a:
      if len(i)>maxlen:
          start = np.random.randint(0, len(i)-maxlen+1)
          b.append(i[start: start+maxlen])
      else:
          b.append(i)
   return b

### 'N': 12 means N is the 12th position in this vector table
index={'pad': 0,  'N': 12, 'D': 9, 'H': 17, 'K': 7, 'I': 6, 'W': 19, 'P': 11, 'Z': 23, 'M': 16, 'Q': 13, 'V': 3, 'X': 20, 'R': 8, 'S': 5, 'T': 10, 'O': 24, 'Y': 15, 'B': 22, 'F': 14, 'G': 2, 'A': 1, 'E': 4, 'L': 0, 'U': 21, 'C': 18}
def to_int(seq, hparam):
	vec=[]
	for i in seq:
		vec.append(index[i])
	for i in range(len(vec), hparam['MAXLEN']):
		vec.append(0)	
	return np.array(vec)
def go_protein_similarity(seq_embed, hparams):
	#Characteristics of GO terms read from a file
	sparse_matrix = np.load('ccCAFA3weight80.npy')
	sparse_matrix *= hparams['hidden_size'] ** 0.5
	embeddings = tf.constant(np.array(sparse_matrix))
	embeddings = tf.cast(embeddings, tf.float32)
	label_embed = tf.expand_dims(embeddings, axis=0)
	label_embed = tf.tile(label_embed, [hparams['batch_size'],1,1])
	similarity_matrix = tf.matmul(seq_embed, label_embed, transpose_b=True)
	similarity_matrix = tf.nn.softmax(similarity_matrix, axis=-1)
	w  = Conv1D(filters=hparams['joint_similarity_filter_size'],kernel_size=hparams['joint_similarity_kernel_size'],\
	 strides=1, padding='same', activation='relu')(similarity_matrix)
	w1 = tf.math.reduce_max(w, axis=-1)
	print(similarity_matrix)
	w1 = tf.nn.softmax(w1, axis=-1)
	w1 = tf.expand_dims(w1, axis=1)
	w2 = tf.matmul(w1,seq_embed)
	return tf.squeeze(w2, axis=1) 

class DALTGO_model(object):

	def __init__(self, hparams):
		self.hparams = hparams	
    #Convert the input sequence into a corresponding embedding representation and add position encode
	def Embedding(self, x):
		hparams=self.hparams
		self.embedding_layer = embedding_layer.EmbeddingSharedWeights(
					hparams["vocab_size"], hparams["hidden_size"])
		embedded_inputs = self.embedding_layer(x)
		lstm_layer = Bidirectional(LSTM(units=40, return_sequences=True))
		with tf.name_scope("add_pos_encoding"):
			length = tf.shape(embedded_inputs)[1]
			pos_encoding = model_utils.get_position_encoding(
				length, hparams["hidden_size"])
			encoder_inputs = embedded_inputs + pos_encoding
			encoder_inputs = lstm_layer(encoder_inputs)
		if self.hparams['train']:
				encoder_inputs = tf.nn.dropout(
					encoder_inputs, rate=self.hparams["layer_postprocess_dropout"])
		self.inputs_padding = model_utils.get_padding(x)
		self.attention_bias = model_utils.get_padding_bias(x)
		return encoder_inputs

	# Transfomer-based sequence encoder
	def Encoder(self, encoder_inputs):
		hparams=self.hparams
		self.encoder_stack = transformer_encode.EncoderStack(hparams)
		return self.encoder_stack(encoder_inputs, self.attention_bias, self.inputs_padding)

	def Main_model(self):
		hparams=self.hparams
		tf.compat.v1.disable_eager_execution()
		inputs = tf.compat.v1.placeholder(shape=(self.hparams['batch_size'], self.hparams['MAXLEN']), dtype=tf.int32)
		outs = tf.compat.v1.placeholder(shape=(self.hparams['batch_size'], self.hparams['nb_classes']), dtype=tf.int32)
		return_box = [inputs, outs]
		encoder_inputs = self.Embedding(inputs)
		encoder_outputs = self.Encoder(encoder_inputs)
		def output_layer(input):
			# argv the input is a tensor with shape [batch, length, hidden_size]
			out1 = tf.keras.layers.MaxPool1D(8, data_format='channels_first')(input)
			out1 = tf.reshape(out1, [hparams['batch_size'], -1])
			out2 = tf.compat.v1.layers.Dense(hparams['nb_classes'], activation='sigmoid', name='dense_out')(out1)
			return out2

		if (hparams['label_embed']==False):
				#Corresponds to baseline model M2
				pred_out = output_layer(encoder_outputs)
				loss = self.loss(outs, pred_out, 'bc')
				return_box.append(loss)
				return_box.append(pred_out)
		else:#Corresponds to baseline model M3
				out1 = go_protein_similarity(encoder_outputs,hparams)
				pred_out=tf.compat.v1.layers.Dense(hparams['nb_classes'], activation='sigmoid', name='dense_out')(out1)
				loss = self.loss(outs, pred_out, 'bc')
				return_box.append(loss)
				return_box.append(pred_out)
		if len (return_box)<5:
			return_box.append(tf.constant([0]))
		return return_box

	def loss(self, ytrue, ypred, loss_type):
		if loss_type == 'bc':
			bce=tf.keras.losses.BinaryCrossentropy()
			return bce(ytrue, ypred)

	def train(self):
		hparams=self.hparams
		data = self.data_load(self.hparams['data_path'])
		def sparse_to_dense(y,  length):
			out=np.zeros((len(y), length), dtype=np.int32)
			for i in range(len(y)):
				#print (y[i])
				for j in y[i]:
					out[i][j]=1
			return out
			
		with tf.device('/gpu:0'):
			holder_list = self.Main_model()  #------------holder_list: [model_input, model_output, loss]
			optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.hparams['lr'])
			train_op = optimizer.minimize(holder_list[2])
			init_op = tf.compat.v1.global_variables_initializer()
		batch_size = self.hparams['batch_size']
		train_x = data[0]
		train_y  = sparse_to_dense(data[1] ,hparams['nb_classes'])
		val_x = data[2]
		val_y= sparse_to_dense(data[3], hparams['nb_classes'])
		val_list=[v for v in tf.compat.v1.global_variables()]
		saver = tf.compat.v1.train.Saver(val_list, max_to_keep=None)
		print ('start training. training information:')
		with tf.compat.v1.Session() as sess:
			sess.run(init_op)
			resume_epoch = 0			
			for epoch in range(hparams['epochs']):
				sepoch_train_loss = 0.
				iterations = int((len(train_x)) // hparams['batch_size'])
				print ("epoch %d begins:" %(resume_epoch+epoch+1))
				print ("#iterations:", iterations)
				for ite in range(iterations):
					x = cut(train_x[ite*batch_size: (ite+1)*batch_size], hparams['MAXLEN'])
					y = train_y[ite*batch_size: (ite+1)*batch_size]
					train_loss ,_ , regular_loss= sess.run([holder_list[2], train_op, holder_list[4]], {holder_list[0]: x, holder_list[1]: y})
					sepoch_train_loss+=train_loss
					print ("iteration %d/%d totaltrain_loss: %.3f" %(ite+1, iterations, train_loss))
				sepoch_train_loss /= iterations
				train_z = list(zip(train_x, train_y))
				shuffle(train_z)
				train_x, train_y = zip(*train_z)
			#evaluation
			epoch = hparams['epochs']-1
			pred_scores=[]
			sepoch_val_loss = 0.
			iterations = int(len(val_x) // hparams['batch_size'])
			for ite in range(iterations):
				x= val_x[ite*batch_size: (ite+1)*batch_size]
				y= val_y[ite*batch_size: (ite+1)*batch_size]
				val_loss, pred_score = sess.run([holder_list[2], holder_list[3]], {holder_list[0]: x, holder_list[1]: y})
				sepoch_val_loss+=val_loss
				pred_scores.extend(pred_score)
				print ("iteration %d/%d val_loss: %.3f" %(ite+1, iterations, val_loss))
			sepoch_val_loss/=iterations
			fmax, smin, auprc,auroc_macro,auroc_micro = metric.main(val_y[:len(pred_scores)], pred_scores, hparams)
			print(fmax)
			print(smin)
			print(auprc)
			print(auroc_micro)
			print(auroc_macro)
			print(" %.3f %.3f %.3f\n" %(fmax, smin, auprc))	
			with open("result.csv", "a") as f:
				f.write("%d %.3f %.3f %.3f %.3f%.3f %.3f\n" %(epoch+resume_epoch+1, sepoch_val_loss, fmax, smin, auprc,auroc_macro,auroc_micro))		
					
	def data_load(self, path):
		with open(path+"/train_seq_"+hparams['ontology'], "rb") as f:
			train_seq = pickle.load(f)
		with open(path+"/train_label_"+hparams['ontology'], "rb") as f:
			train_label = pickle.load(f)
		with open(path+"/test_seq_"+hparams['ontology'], "rb") as f:
			test_seq = pickle.load(f)
		with open(path+"/test_label_"+hparams['ontology'], "rb") as f:
			test_label = pickle.load(f)
		train_X=[]
		train_Y= train_label
		test_X=[]
		test_Y= test_label
		for i in range(len(train_seq)):
			train_X.append(to_int(train_seq[i]['seq'], self.hparams))
		for i in range(len(test_seq)):
			test_X.append(to_int(test_seq[i]['seq'], self.hparams))
		#Select the return data based on the training mode
		return train_X, train_Y, test_X, test_Y

if __name__== "__main__":
	flags = tf.compat.v1.flags
	flags.DEFINE_integer("batch_size", 32, "e")
	flags.DEFINE_integer("epochs", 30, "e")
	flags.DEFINE_float("lr", 1e-3, "e")
	flags.DEFINE_string("save_path", './', "model savepath")
	flags.DEFINE_string("ontology", 'cc', "e")
	flags.DEFINE_integer("nb_classes", None, "e")
	flags.DEFINE_bool("label_embed", True, "e")
	flags.DEFINE_string("data_path", '../data/CAFA3/', "path_to_store_data")
	flags.DEFINE_float("regular_lambda", 0, "e")
	#the number of transformer attention head
	flags.DEFINE_integer("num_heads", 2, "e")
	flags.DEFINE_integer("num_hidden_layers", 6, "e")
	#the dimention of vector
	flags.DEFINE_integer("hidden_size", 80, "e")
	FLAGS = flags.FLAGS
	hparams = hparam.params(flags)
	model1 = DALTGO_model(hparams)
	model1.train()

