import tensorflow as tf
import pickle
import sys
def params(flags):
	hparams={}
	FLAGS = flags.FLAGS
	hparams['MAXLEN'] = 1000
	hparams['batch_size'] = FLAGS.batch_size
	hparams['epochs'] = FLAGS.epochs
	hparams['lr'] = FLAGS.lr
	hparams['save_path'] = FLAGS.save_path	
	hparams['ontology'] = FLAGS.ontology
	hparams['data_path'] = FLAGS.data_path
	#the number of GO classes
	hparams['nb_classes']=6681
	flags.DEFINE_integer("vocab_size", 26, "e")
	hparams['vocab_size'] = FLAGS.vocab_size
	#---------------------------------------------------Transformer Params:
	hparams['hidden_size'] = FLAGS.hidden_size
	hparams['num_hidden_layers'] = FLAGS.num_hidden_layers
	hparams['num_heads'] = FLAGS.num_heads
	flags.DEFINE_bool("train", True, "e")
	hparams['train'] = FLAGS.train
	hparams['joint_similarity_kernel_size'] = 10
	hparams['joint_similarity_filter_size'] = 10
	hparams['regular_lambda'] = FLAGS.regular_lambda
	hparams['tsnet_filter'] = [10, 5]
	hparams['tsnet_kernel'] = [10, 5]
	hparams['tsnet_stride'] = [5, 2]
	hparams['tsnet_pool'] = [5, 2] 
	hparams['label_embed'] = FLAGS.label_embed
	hparams['layer_postprocess_dropout']=0.1
	hparams['attention_dropout']=0.1
	hparams['relu_dropout']=0.1
	hparams['filter_size'] =64
	# TPU 
	hparams['use_tpu']  = False
	hparams['allow_ffn_pad']=True
	return  hparams

if __name__== "__main__":


	flags = tf.flags
	#--------------------------------------------------training HParams:
	flags.DEFINE_string("main_model", "SALT", "e")
	flags.DEFINE_integer("batch_size", 256, "e")
	flags.DEFINE_integer("epochs", 20, "e")
	flags.DEFINE_float("lr", 1e-5, "e")
	flags.DEFINE_string("save_path", '../trained_models/', "model savepath")
	flags.DEFINE_string("resume_model", None, "")
	flags.DEFINE_string("ontology", 'cc', "e")
	flags.DEFINE_integer("nb_classes", 1070, "e")
	flags.DEFINE_bool("label_embed", False, "e")
	hparams = params(flags)

	with open (sys.argv[1]+".hparam", "wb") as f:	
		pickle.dump(hparams, f)	
