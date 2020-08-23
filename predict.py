import numpy as np
import math
import cv2
import os
import json
num =0
result = []
count = 0
mess_temp = "stand"
mode = 0
#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from ...utils.box import BoundBox
from ...cython_utils.cy_yolo2_findboxes import box_constructor
def expit(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def findboxes(self, net_out):
	# meta
	meta = self.meta
	boxes = list()
	boxes=box_constructor(meta,net_out)
	return boxes

def postprocess(self, net_out, im, save = True):
	global num
	global result
	global count
	global mess_temp
	global mode
	"""
	Takes net output, draw net_out, save to disk
	"""

	boxes = self.findboxes(net_out)
	# meta
	meta = self.meta
	threshold = meta['thresh']
	colors = meta['colors']
	labels = meta['labels']
	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im
	h, w, _ = imgcv.shape
	
	resultsForJSON = []
	for b in boxes:
		boxResults = self.process_box(b, h, w, threshold)
		if boxResults is None:
			continue
		left, right, top, bot, mess, max_indx, confidence = boxResults
		thick = int((h + w) // 300)
		if self.FLAGS.json:
			resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
			continue

		cv2.rectangle(imgcv,
			(left, top), (right, bot),
			colors[max_indx], thick)
		cv2.putText(imgcv, mess, (left, top - 12),
			0, 1e-3 * h, colors[max_indx],thick//3)

		if num == 0:
			if mess == "stand" :
				count = count+1
				if count > 10:
					num = num+1
					result.append(mess)
					print(result)
					#print(num)
					count = 0
			else :
				count = 0

		else :
			if result[num-1] != mess:
				if mess_temp == mess:
					count = count+1
					if count > 10:
						result.append(mess)
						num = num+1
						count = 0
						print(result)
				else :
					count = 0
			if len(result) > 3 and mess == "stand" :
				# print(result)
				if result[0] == "stand" and result[1] == "left1" and result[2] == "left2" and result[3] == "left3":
					result = []
					num = 0
					mode = 1
					print("왼쪽에서 오른쪽으로")
				elif result[0] == "stand" and result[1] == "right1" and result[2] == "right2" and result[3] == "right3":
					result = []
					num = 0
					mode = 2
					print("오른쪽에서 왼쪽으로")
				elif result[0] == "stand" and result[1] == "stop1" and result[2] == "stop2":
					result = []
					num = 0
					mode = 3
					print("STOP")

		if ( mode == 1):
			cv2.putText(imgcv, '====>', (850, 300), 0, 2, (0, 0, 255), thickness =3)
		elif (mode ==2):
			cv2.putText(imgcv, '<====', (200, 300), 0, 2, (0, 0, 255), thickness = 3)
		elif (mode == 3):
			cv2.putText(imgcv, 'STOP', (580, 70), 0, 2, (0, 0, 255), thickness =3)
		mess_temp = mess






		# if mess == "stand" :
		# 	print(mess + "------------=====================================")

		# print(mess + "------------------------------------------------")

	if not save: return imgcv

	outfolder = os.path.join(self.FLAGS.imgdir, 'out')
	img_name = os.path.join(outfolder, os.path.basename(im))
	if self.FLAGS.json:
		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.splitext(img_name)[0] + ".json"
		with open(textFile, 'w') as f:
			f.write(textJSON)
		return

	cv2.imwrite(img_name, imgcv)
