import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from utils.torch_utils import initialize_weights

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for mixed precision and faster training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

wdir = 'weights' + os.sep  # weights dir
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'

# Hyperparameters https://github.com/ultralytics/yolov3/issues/310

hyp = {# 'giou': 3.54,  # giou loss gain
       'giou': 2.54,  # trash giou loss gain （4.54）太大直接不收敛
       #'giou': 1.58,  # evolved haihua giou loss gain
       #'cls': 37.4,  # coco original cls loss gain
       'cls': 156.6, # haihua cls loss gain
       #'cls': 197.0,  # evolved haihua cls loss gain
       #'cls_pw': 0.95,  # cls BCELoss positive_weight
       'cls_pw': 1.06,  # evolved haihua cls BCELoss positive_weight
       # 'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       #'obj': 154.2,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj': 164.0,  # evolved haihua obj loss gain
       #'obj_pw': 0.85,  # obj BCELoss positive_weight
       'obj_pw': 0.78,  # eovlved haihua obj BCELoss positive_weight
       #'iou_t': 0.225,  # iou training threshold
       'iou_t': 0.19,  # evolved haihua iou training threshold 
       # 'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       #'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lr0': 0.00958,  # evolved haihua initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       # 'lrf': 0.0005,  # final learning rate (with cos scheduler)
       #'momentum': 0.937,  # SGD momentum
       'momentum': 0.947,  # evolved haihua SGD momentum
       #'weight_decay': 0.000584,  # optimizer weight decay
       'weight_decay': 0.000545,  # evolved haihua optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0171,  # evolved haihua image HSV-Hue augmentation (fraction)
       #'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.714,  # evolved haihua image HSV-Saturation augmentation (fraction)
       #'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.277,  # evolved haihua image HSV-Value augmentation (fraction)
       #'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt (optional)  # hyp*.txt在哪？？
f = glob.glob('hyp*.txt')  # TODO 这里的操作在哪？  在指令形式下运行，会运行这里吗？
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v

# Print focal loss if gamma > 0
if hyp['fl_gamma']:
    print('Using FocalLoss(gamma=%g)' % hyp['fl_gamma'])


def train(hyp):
    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt.weights  # initial training weights
    imgsz_min, imgsz_max, imgsz_test = opt.img_size  # img sizes (min, max, test)  # TODO 这里的opt.img_size
    # TODO 可以设置成默认的列表[,,]里面的元素分别对应[min_train, max-train, test] image size

    # Image Sizes
    gs = 64  # (pixels) grid size  # TODO 这里需要小改(2020年 4月28 看完感觉不需要)
    assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)  # TODO 这里设置的gs=64是为何？
    opt.multi_scale |= imgsz_min != imgsz_max  # multi if different (min, max)  # 不明白！！ # TODO 异或？？
    if opt.multi_scale:  # TODO gs是为了多尺度训练使用的
        if imgsz_min == imgsz_max:
            imgsz_min //= 1.5
            imgsz_max //= 0.667
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs  # TODO floor操作
        imgsz_min, imgsz_max = grid_min * gs, grid_max * gs
    img_size = imgsz_max  # initialize with max size  #

    # Configure run
    init_seeds()
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = 1 if opt.single_cls else int(data_dict['classes'])  # number of classes
    hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset

    # Remove previous results
    for f in glob.glob('*_batch*.png') + glob.glob(results_file):
        os.remove(f)

    # Initialize model
    model = Darknet(cfg).to(device)
    # model = initialize_weights(model)  # 模型初始化-->ok，解决

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:  # TODO BatchNorm2d也包含'.bias', k 获得的是一个字符串,'.bias'在'BatchNorm2d.bias'中
            pg2 += [v]  # biases  # TODO pg2存储全部的bias
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
            # TODO：pg1存储所有的卷积层的weight
        else:
            pg0 += [v]  # all else  # TODO pg0存储的不知道是什么

    if opt.adam:
        # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])  # TODO bias的学习率为0.01
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)  # TODO 加权移动平均
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    # TODO gp1(卷积层的权重)使用L2(weight_deacy)正则化
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    # len(optimizer.param_groups)=3,这样是为了能将不同层区分调参吗
    del pg0, pg1, pg2
    # TODO：继承自torch.optim.Optimizer类的优化器如Adam、SGD等一般具有两个属性：
    #  ①：optimizer.default:   字典(是字典！！！！)
    #  存放这个优化器的一些初始参数，有：'lr', 'betas', 'eps', 'weight_decay', 'amsgrad'。
    #  事实上这个属性继承自torch.optim.Optimizer父类；
    #  ②：optimizer.param_groups:  列表(是列表！！！)每个元素都是一个字典，每个元素包含的关键字有：'params', 'lr',
    #  'betas', 'eps', 'weight_decay', 'amsgrad'，params是各个网络的参数放在了一起(详细见下面的注释)。
    #  这个属性也继承自torch.optim.Optimizer父类。
    #
    #  TODO：由于上述两个属性都继承自所有优化器共同的基类，所以是所有优化器类都有的属性，并且两者字典中键名相同的元素
    #   值也相同（经过lr_scheduler后lr就不同了）

    '''
        假设
        net1= module(), net2 = module()
        optimizer1 = torch.optim.Adam([*net1.parameters, *net2.parameter], lr = float),
        该形式将net1.parameter和net2.parameter以tuple形式存进一组优化器的参数中，len(optimizer1.para_groups)=1（因为是组合成一组）
        
        optimizer2 = torch.optim.Adam([{'params':net1.parameters()},{'params':net1.parameters()}], lr = float)
        该形式是将net1和net2的参数分别存进Adam优化器，len(optimizer2.param_groups)=2,
        optimizer.param_groups[0]存放net1的网络参数，optimizer.param_groups[1]存放net2的网络参数
    '''
    # TODO 注意：lr_scheduler更新optimizer的lr，是更新的optimizer.param_groups[n][‘lr’]，而不是optimizer.defaults[‘lr’]
    start_epoch = 0
    best_fitness = 0.0
    attempt_download(weights)  # 不会下载（可注释掉）
    if weights.endswith('.pt'):  # pytorch format
        # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        chkpt = torch.load(weights, map_location=device)
        # TODO 这里可以使用迁移学习,具体代码为:model.load_state_dict(torch.load(PATH), strict=False)

        # load model
        try:
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
            # TODO 这里已经令load_state_dict()中的strict参数设置为False了,似乎还是会报错(对于修改了
            #  backbone的模型，仍然只能使用scratch训练)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        # load optimizer
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

        # load results
        if chkpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        start_epoch = chkpt['epoch'] + 1
        del chkpt

    elif len(weights) > 0:  # darknet format  TODO:如果是使用自己的'*.weights'文件,直接在这里加载模型的预训练参数
        # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        load_darknet_weights(model, weights)

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Scheduler https://github.com/ultralytics/yolov3/issues/238
    lf = lambda x: (((1 + math.cos(
        x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine https://arxiv.org/pdf/1812.01187.pdf
    # TODO lf()送入的实参是epoch--->lf(epoch)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch - 1)
    # TODO 当optimizer作为参数传递给lr_scheduler时,lr_scheduler会给optimizer对应group(字典列表)生成新的键值对,key为'initial_lr',值=对应group的
    #  ‘optimizer.default['lr']’！！！！！！！！！！！！！！！！！！！！！！！
    #
    # scheduler = lr_scheduler.MultiStepLR(optimizer, [round(epochs * x) for x in [0.8, 0.9]], 0.1, start_epoch - 1)
    #
    # TODO：lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1),其中：
    #  optimizer是优化器(需要更改学习率的优化器);
    #  milestone(list)：递增list，存放要更新lr的epoch;
    #  获得的新的学习率 new_lr = initial_lr × lr_lambda(γ是python的排序模块bisect中的bisect_right(milestone,epoch)函数获得的γ,简单来说就是
    #  在epoch到了升序排列的milestone列表中元素的时候，让initial_lr×lr_lambda。
    #  last_epoch=-1表示从头开始训练，即从epoch=1开始。
    #
    # TODO: lr.scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=stare_epoch-1),
    #  更新策略： new_lr = initial_lr * lambda, lambda是通过上面的lf(epoch)函数获得的
    # Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, '.-', label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataset
    dataset = LoadImagesAndLabels(train_path, img_size, batch_size,  # img_size为指令设置的图片最大长度
                                  augment=True,  # TODO 这里需要看一看是否需要augment（coco可能需要,海华垃圾可能也需要）
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training # TODO 默认没有在训练时使用rect
                                  cache_images=opt.cache_images,
                                  single_cls=opt.single_cls)

    # Dataloader
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Testloader
    testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                                                 hyp=hyp,
                                                                 rect=True,
                                                                 cache_images=opt.cache_images,
                                                                 single_cls=opt.single_cls),
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Model parameters
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)  TODO 这里训练的时候也是需要更改的？？
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    # TODO 这个labels_to_class_weights()现在没有用，后面再看

    # Model EMA
    ema = torch_utils.ModelEMA(model)

    # Start training
    nb = len(dataloader)  # number of batches
    n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
    maps = np.zeros(nc)  # mAP per class
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    print('Image sizes %g - %g train, %g test' % (imgsz_min, imgsz_max, imgsz_test))
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()  # 模型设置为train模式

        # Update image weights (optional)  # TODO 待看
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(4).to(device)  # mean losses
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            # LoadImagesAndLabels类通过dataloader返回batch_size的(imgs,targets,paths(图片存放路径),shapes)
            ni = i + nb * epoch  # number integrated batches (since train start)
            # TODO ni是训练的批次数量，加入batch数量是300个，一个epoch完，训练的batch数量就是 nb * epoch
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            # Burn-in
            # if ni <= n_burn * 2:
            if ni <= n_burn:  # TODO 上面这一行改成下面的if
                xi = [0, n_burn]  # x interp  # TODO 多的一行 添加了xi
                # model.gr = np.interp(ni, [0, n_burn * 2], [0.0, 1.0])  # TODO 这一行被注释掉了
                model.gr = np.interp(ni, xi, [0.0, 1.0])  # TODO 改行代替上一行的model.gr
                accumulate = max(1, np.interp(ni, xi, [1, 64/batch_size]).round())  # TODO 添加了accumulate
                # giou loss ratio (obj_loss = 1.0 or giou)  # TODO 不懂(obj loss的惩罚因子1和giou区别)
                # TODO np.interp(x(插值横坐标值), xp(横坐标范围,为列表,列表元素表示点的个数), fp(纵坐标范围,列表,列表元素表示点的个数))线性插值；
                #  这里线性插值是返回横坐标轴上x[0, n_burn * 2]区间,纵坐标轴y[0, 1]的两个点之间插入x=ni时的纵轴y的坐标；
                #  但是这里和模型的giou loss ratio有什么关系？？？？

                # TODO 2020年7月23日 comment  这部分被注释掉
                # if ni == n_burn:  # burnin complete
                #     print_model_biases(model)

                # TODO 动态调整前几个epoch的学习率
                for j, x in enumerate(optimizer.param_groups):  # TODO optimizer.param_groups有三组param---
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    # x['lr'] = np.interp(ni, [0, n_burn], [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])  # TODO 原先的
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])  # TODO 使用xi代替[0, n_burn]
                    x['weight_decay'] = np.interp(ni, xi, [0.0, hyp['weight_decay'] if j == 1 else 0.0])  # TODO 多添加了'weight_deacy'
                    if 'momentum' in x:
                        # x['momentum'] = np.interp(ni, [0, n_burn], [0.9, hyp['momentum']])  # TODO hyp['momentum']=0.937
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])  # TODO 使用xi代替上面的[0, n_burn]

            # Multi-Scale training
            if opt.multi_scale:
                if ni / accumulate % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
                    img_size = random.randrange(grid_min, grid_max + 1) * gs
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Run model
            pred = model(imgs)

            # Compute loss  TODO loss后面再改吧
            loss, loss_items = compute_loss(pred, targets, model)  # 这里需要选择合适的target，而不是全部的target
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Scale loss by nominal batch_size of 64
            loss *= batch_size / 64  # TODO 这里需要更改

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # TODO 为什么ni % accumulate==0计算梯度并更新参数
            # Optimize accumulated gradient
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)  # TODO 将model.state_dict()返回的参数(网络中只有卷积层、线性层等具有可学习参数的层有state_dict条目)
                # TODO 字典的值进行指数加权平均 v(t) = α*v(t-1) + (1-α)*weight(t),v(t)是t时刻的shadow weights

            # Print batch results  # TODO
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
            pbar.set_description(s)

            # Plot images with bounding boxes
            if ni < 1:
                f = 'train_batch%g.png' % i  # filename
                plot_images(imgs=imgs, targets=targets, paths=paths, fname=f)
                if tb_writer:
                    tb_writer.add_image(f, cv2.imread(f)[:, :, ::-1], dataformats='HWC')
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        scheduler.step()  # TODO scheduler.step()对模型优化器的学习率进行调整

        # Process epoch results
        ema.update_attr(model)  # TODO 这里难道不应该
        final_epoch = epoch + 1 == epochs
        if not opt.notest or final_epoch:  # Calculate mAP
            is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
            results, maps = test.test(cfg,
                                      data,
                                      batch_size=batch_size,
                                      img_size=imgsz_test,  # TODO imgsz_test使用opt指令指定数值
                                      model=ema.ema,  # TODO ema.ema=deepcopy(model)
                                      save_json=final_epoch and is_coco,
                                      single_cls=opt.single_cls,
                                      dataloader=testloader,
                                      multi_label=ni > n_burn)  # TODO testloader里rect=True
            # TODO results=(mp, mr, map, mf1, *(loss.cpu() / len(dataloader)), maps=maps?????
        # Write epoch results
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        if len(opt.name) and opt.bucket:
            os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (opt.bucket, opt.name))

        # Write Tensorboard results
        if tb_writer:
            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/F1',
                    'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
            for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                tb_writer.add_scalar(tag, x, epoch)

        # Update best mAP # TODO (平均精度均值)
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save training results
        save = (not opt.nosave) or (final_epoch and not opt.evolve)
        if save:
            '''
                这里到torch.save(chkpt)的操作和存储想存储的模型参数操作相同:
                ①：定义chkpt={'key': value}
                ②：torch(chkpt, '目标路径')
            '''
            with open(results_file, 'r') as f:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': f.read(),
                         'model': ema.ema.module.state_dict() if hasattr(model, 'module') else ema.ema.state_dict(),
                         'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last checkpoint
            torch.save(chkpt, last)  # TODO 自定义数据集的话，在nosave情况下，会存储模型训练结果到'./weights/last.pt'

            # 这里应该可以存储模型参数成为.weights文件
            cfg_name = str(cfg[4:-4])
            save_weights(model, path='./weights/%s.weights' % cfg_name)

            # Save best checkpoint
            if (best_fitness == fi) and not final_epoch:
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            # if epoch > 0 and epoch % 10 == 0:
            #     torch.save(chkpt, wdir + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt

        # end epoch ----------------------------------------------------------------------------------------------------

    # end training
    n = opt.name
    if len(n):
        n = '_' + n if not n.isnumeric() else n
        fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
        for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # rename
                ispt = f2.endswith('.pt')  # is *.pt
                strip_optimizer(f2) if ispt else None  # strip optimizer
                os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload

    if not opt.evolve:
        plot_results()  # save as results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=8)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--accumulate', type=int, default=2, help='batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/template.data', help='*.data path')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', nargs='+', type=int, default=[768, 768, 768], help='[min_train, max-train, test] img sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    opt = parser.parse_args()
    opt.weights = last if opt.resume else opt.weights
    print(opt)
    opt.img_size.extend([opt.img_size[-1]] * (3 - len(opt.img_size)))  # extend to 3 sizes (min, max, test)
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)  # TODO 这里指定device
    if device.type == 'cpu':
        mixed_precision = False

    # scale hyp['obj'] by img_size (evolved at 320)
    # hyp['obj'] *= opt.img_size[0] / 320.

    tb_writer = None
    if not opt.evolve:  # Train normally
        try:
            # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter()
            print("Run 'tensorboard --logdir=runs' to view tensorboard at http://localhost:6006/")
        except:
            pass

        train(hyp)  # train normally

    else:  # Evolve hyperparameters (optional)
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(1):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                method, mp, s = 3, 0.9, 0.2  # method, mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # gains
                ng = len(g)
                if method == 1:
                    v = (npr.randn(ng) * npr.random() * g * s + 1) ** 2.0
                elif method == 2:
                    v = (npr.randn(ng) * npr.random(ng) * g * s + 1) ** 2.0
                elif method == 3:
                    v = np.ones(ng)
                    while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                        # v = (g * (npr.random(ng) < mp) * npr.randn(ng) * s + 1) ** 2.0
                        v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = x[i + 7] * v[i]  # mutate

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Train mutation
            results = train(hyp.copy())  # TODO 原始为没有hyp.copy()

            # Write mutation results
            print_mutation(hyp, results, opt.bucket)

            # Plot results
            plot_evolution_results(hyp)
