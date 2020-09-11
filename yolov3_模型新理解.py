# YOLOv3新学习获得知识点
# 1. compute_loss和build_targets函数的新认识
def build_targets(p, targets, model):
	"""
		1.获得anchors
		2.计算anchors与ground_truth box的IoU（交并比）
		3.选择使用all_anchor还是max_iou的anchor
		4.进行匹配寻找正例
		5.返回正例样本的相关索引、类别，与正例样本的box信息和anchor信息
	"""
	indices = [],
	gain = torch.ones(6, dtype=torch.int)
	nt = targets.shape[0]
	# 获得anchors
	for i, j in enumerate(model.yolo_layers):  # model.yolo_layers保存的是yolo层在
	    # module_list中的index
		anchors = model.module_list[j].anchor_vec  # 然后model.module_list[j]获得的是存在
		# module_list中的各Yolo类模块，Yolo类模块中定义了self.anchor_vec的列表
		na = ancors.shape[0]
		# gain，gain主要是对targets中的xywh的尺寸进行修改
		gain[2:] = torch.tensor(p[i].shape)[[3,2,3,2]]
		t, a = targets * gain, [] # t变成输出尺寸下的target位置
		gwh = t[:, 4:6] 
		# 计算IoU
		iou = box_IoU(anchor, gwh)
		
		use_all = True, reject = True
		
		# get indses
		if use_all:
			a = torch.range(na).view(-1, 1).repeat(nt).view(-1)  # 先构建好所有的anchor的索引，shape(nt*m)
			t = t.repeat(na, 1)  # t的shape(nt*m, 6)
		else:
			iou, a = iou.max(0)  # iou和a都是形状(nt,)的, 不过a中的元素是返回最大anchor与target的wh iou的anchor索引
			
		if rejecct:
			j = iou.view(-1) > model.hpy['iou_thresold']  # j 形状：当use_all为True时为（m*nt,）,否则(nt,),j元素为bool性
			t, a = t[j], a[j]  # 通过j的bool型选择正例t和与正例对应的wh_IoU(anchor,gwh)大于iou_thresold的indices
		
	b, c = t[:, 0:2].long().t()  # 获得正例
	gxy = t[:, 2:4]
	gwh = t[:, 4:]
	gi, gj = gxy
	indices.append(b, a, gj, gi)
	
	
		

def box_IoU(box1, box2):
	box1 = box1[:, None]  # 为box1在dim1增加一个维度——box1代表anchor,shape = (m, 2)
	box2 = box2[None]
	inter = torch.min(box1, box2).prod(2)  # box1和box2分别在dim2求prod，元素与元素想乘
	return  inter / (box1.prod(2) + box2.prod(2) - inter)