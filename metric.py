import sklearn
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
import numpy as np
def auprc(ytrue, ypred):
  p, r, t =  precision_recall_curve(ytrue, ypred)
  return auc(r,p)

def main(ytrue1, ypred1, hparams=None):
	print(ytrue1)
	fmax = 0
	smin=100000.0
	prec_list = []
	recall_list=[]
	ic = np.load('../data/ours/bp-ic.npy')
	ytrue=[]
	ypred=[]
	# delete those sample whose labels are all 0.
	for i in range(len(ytrue1)):
		if np.sum(ytrue1[i]) >0:
			ytrue.append(ytrue1[i])
			ypred.append(ypred1[i])	
	for t in range(1, 101):
		thres = t/100
		thres_array=np.ones((len(ytrue), len(ytrue[0])), dtype=np.float32) * thres
		pred_labels = np.greater(ypred, thres_array).astype(int)
		tp_matrix =pred_labels*ytrue
		tp = np.sum(tp_matrix, axis=1, dtype=np.int32)
		tpfp = np.sum(pred_labels, axis=1)
		tpfn = np.sum(ytrue,axis=1)
		#---------computing s1 and update smin if possible
		ru = ytrue- tp_matrix
		mi = pred_labels - tp_matrix
		ruu = np.mean( np.matmul(ru, ic.astype(np.float32)))
		mii = np.mean( np.matmul(mi, ic.astype(np.float32)))
		s1 = np.sqrt(ruu**2 +mii**2)
		smin=min(smin, s1)
		print(smin)
		#---------computing f1 and update fmax if possible
		avgprs=[]
		for i in range(len(tp)):
			if tpfp[i]!=0:
				avgprs.append(tp[i]/float(tpfp[i]))

		if len(avgprs)==0:
			continue
		avgpr = np.mean(avgprs)
		avgrc = np.mean(tp/tpfn)
		prec_list.append(avgpr)
		recall_list.append(avgrc)
		f1 = 2*avgpr*avgrc/(avgpr+avgrc)
		fmax=max(fmax, f1)
		auroc_micro = roc_auc_score(np.array(ytrue).flatten(), np.array(ypred).flatten(), multi_class='ovr', average='micro')
		auroc_macro = roc_auc_score(np.array(ytrue).flatten(), np.array(ypred).flatten(), multi_class='ovr', average='macro')
	return fmax, smin, auprc(np.array(ytrue).flatten(), np.array(ypred).flatten()),auroc_macro,auroc_micro

	
if __name__=='__main__':
	ytrue=np.array([[0,1,1,0], [1,0,1,0]])
	ypred = np.array([[0.2, 0.3, 0.3,0.5], [ 0.2, 0.3, 0.5, 0.3] ])
	hparams={'data_path' : '../../data//', 'ontology':"cc"}
	main(ytrue, ypred, hparams)
	print (auprc(ytrue[0], ypred[0]))

	
