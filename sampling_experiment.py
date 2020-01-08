import argparse
import time

import torch.distributed as dist
from torch.utils.data import DataLoader

import test  # Import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--test_only', action='store_true', help='do testing only')
opt = parser.parse_args()

def test(
        cfg,
        data_cfg,
        weights=None,
        batch_size=16,
        img_size=416,
        iou_thres=0.5,
        conf_thres=0.8,
        nms_thres=0.5,
        save_json=False,
        model=None,
        first_img=0,
        last_img=None
):
    Y = 0 # for counting # defects in truth 
    y = 0 # for counting # defects predicted  
    num_correct = 0 # number of predicted defects above iou threshold 
    if model is None:
        device = torch_utils.select_device()

        # Initialize model
        model = Darknet(cfg, img_size).to(device)

        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        device = next(model.parameters()).device  # get model device

    # Configure run
    data_cfg = parse_data_cfg(data_cfg)
    nc = int(data_cfg['classes'])  # number of classes
    test_path = data_cfg['valid']  # path to test images
    names = load_classes(data_cfg['names'])  # class names

    # Dataloader
    dataset = LoadImagesAndLabels(test_path, img_size=img_size,first_img=first_img,last_img=last_img)
    if weights:
        print("num of imgs tested: "+ str(len(dataset)) + " from {0} to {1}".format(first_img, last_img) + " using " + weights)
    else:
        print("num of imgs tested: "+ str(len(dataset)) + " from {0} to {1}".format(first_img, last_img))
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=4,
                            pin_memory=False,
                            collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    coco91class = coco80_to_coco91_class()
    print(('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1'))
    loss, p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc='Computing mAP')):
        targets = targets.to(device)
        imgs = imgs.to(device)

        # Plot images with bounding boxes
        if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
            plot_images(imgs=imgs, targets=targets, fname='test_batch0.jpg')

        # Run model
        inf_out, train_out = model(imgs)  # inference and training outputs

        # Build targets
        target_list = build_targets(model, targets)

        # Compute loss
        loss_i, _ = compute_loss(train_out, target_list)
        loss += loss_i.item()

        # Run NMS
        output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            Y += nl
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(img_size, box, shapes[si])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for di, d in enumerate(pred):
                    jdict.append({
                        'image_id': image_id,
                        'category_id': coco91class[int(d[6])],
                        'bbox': [float3(x) for x in box[di]],
                        'score': float(d[4])
                    })

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            y += len(pred)
            if nl:
                detected = []
                tbox = xywh2xyxy(labels[:, 1:5]) * img_size  # target boxes

                # Search for correct predictions
                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    iou, bi = bbox_iou(pbox, tbox).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and bi not in detected:
                        correct[i] = 1
                        num_correct += 1
                        detected.append(bi)

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))

    # Compute statistics
    stats_np = [np.concatenate(x, 0) for x in list(zip(*stats))]
    nt = np.bincount(stats_np[3].astype(np.int64), minlength=nc)  # number of targets per class
    if len(stats_np):
        p, r, ap, f1, ap_class = ap_per_class(*stats_np)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1), end='\n\n')

    # Print results per class
    if nc > 1 and len(stats_np):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Save JSON
    if save_json and map and len(jdict):
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataset.img_files]
        with open('results.json', 'w') as file:
            json.dump(jdict, file)

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        cocoGt = COCO('../coco/annotations/instances_val2014.json')  # initialize COCO ground truth api
        cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api

        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        map = cocoEval.stats[1]  # update mAP to pycocotools mAP

    # Return results
    return mp, mr, map, mf1, loss, Y, y, num_correct

def train(
        cfg,
        data_cfg,
        img_size=416,
        resume=False,
        epochs=273,  # 500200 batches at bs 64, 117263 images = 273 epochs
        batch_size=16,
        accumulate=1,
        multi_scale=False,
        freeze_backbone=False,
        num_workers=4,
        transfer=False,  # Transfer learning (train only YOLO layers)
        first_img=0,
        last_img=None

):
    weights = 'weights' + os.sep
    latest = weights + 'latest.pt'
    best = weights + 'best.pt'
    device = torch_utils.select_device()

    if multi_scale:
        img_size = 608  # initiate with maximum multi_scale size
        num_workers = 0  # bug https://github.com/ultralytics/yolov3/issues/174
    else:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Configure run
    train_path = parse_data_cfg(data_cfg)['train']

    # Initialize model
    model = Darknet(cfg, img_size).to(device)

    # Optimizer
    lr0 = 0.001  # initial learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=0.9, weight_decay=0.0005)

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    best_map = 0
    yl = get_yolo_layers(model)  # yolo layers
    nf = int(model.module_defs[yl[0] - 1]['filters'])  # yolo layer size (i.e. 255)

    if resume or transfer:  # Load previously saved model
        if transfer:  # Transfer learning
            chkpt = torch.load(weights + 'yolov3-spp.pt', map_location=device)
            model.load_state_dict({k: v for k, v in chkpt['model'].items() if v.numel() > 1 and v.shape[0] != 255},
                                  strict=False)
            for p in model.parameters():
                p.requires_grad = True if p.shape[0] == nf else False

        else:  # resume from latest.pt
            chkpt = torch.load(latest, map_location=device)  # load checkpoint
            model.load_state_dict(chkpt['model'])

        start_epoch = chkpt['epoch'] + 1
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_loss = chkpt['best_loss']
        # del chkpt

    else:  # Initialize model with backbone (optional)
        if '-tiny.cfg' in cfg:
            cutoff = load_darknet_weights(model, weights + 'yolov3-tiny.conv.15')
        else:
            cutoff = load_darknet_weights(model, weights + 'darknet53.conv.74')

    # Set scheduler (reduce lr at epochs 218, 245, i.e. batches 400k, 450k)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[218, 245], gamma=0.1,
                                                     last_epoch=start_epoch - 1)

    # Dataset
    dataset = LoadImagesAndLabels(train_path, img_size=img_size, augment=True,first_img=first_img, last_img=last_img)

    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend=opt.backend, init_method=opt.dist_url, world_size=opt.world_size, rank=opt.rank)
        model = torch.nn.parallel.DistributedDataParallel(model)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False,
                            pin_memory=False,
                            collate_fn=dataset.collate_fn,
                            sampler=sampler)

    # Start training
    t = time.time()
    model_info(model)
    nB = len(dataloader)
    print("size of data: "+ str(len(dataset)))

    n_burnin = min(round(nB / 5 + 1), 1000)  # burn-in batches
    os.remove('train_batch0.jpg') if os.path.exists('train_batch0.jpg') else None
    os.remove('test_batch0.jpg') if os.path.exists('test_batch0.jpg') else None
    for epoch in range(start_epoch, epochs):
        model.train()
        print(('\n%8s%12s' + '%10s' * 7) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))

        # Update scheduler
        scheduler.step()

        # Freeze backbone at epoch 0, unfreeze at epoch 1
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        mloss = defaultdict(float)  # mean loss
        for i, (imgs, targets, _, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            nt = len(targets)
            if nt == 0:  # if no targets continue
                continue

            # Plot images with bounding boxes
            if epoch == 0 and i == 0:
                plot_images(imgs=imgs, targets=targets, fname='train_batch0.jpg')

            # SGD burn-in
            if epoch == 0 and i <= n_burnin:
                lr = lr0 * (i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr

            # Run model
            pred = model(imgs)

            # Build targets
            target_list = build_targets(model, targets)

            # Compute loss
            loss, loss_dict = compute_loss(pred, target_list)

            # Compute gradient
            loss.backward()

            # Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == nB:
                optimizer.step()
                optimizer.zero_grad()

            # Running epoch-means of tracked metrics
            for key, val in loss_dict.items():
                mloss[key] = (mloss[key] * i + val) / (i + 1)

            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, nB - 1),
                mloss['xy'], mloss['wh'], mloss['conf'], mloss['cls'],
                mloss['total'], nt, time.time() - t)
            t = time.time()
            print(s)

            # Multi-Scale training (320 - 608 pixels) every 10 batches
            if multi_scale and (i + 1) % 10 == 0:
                dataset.img_size = random.choice(range(10, 20)) * 32
                print('multi_scale img_size = %g' % dataset.img_size)

#         Calculate mAP 
        with torch.no_grad():
           results = test(cfg, data_cfg, batch_size=batch_size, img_size=img_size, model=model)

#         Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 5 % results[0:5] + '\n')  # P, R, mAP, F1, test_loss

#         Update best loss
        test_loss = results[4]
        if test_loss < best_loss:
           best_loss = test_loss
        
        # Save training results
        save = True
        if save:
            # Create checkpoint
            chkpt = {'epoch': epoch,
                     'best_loss': best_loss,
                     'model': model.module.state_dict() if type(
                         model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                     'optimizer': optimizer.state_dict()}

            # Save latest checkpoint
            torch.save(chkpt, latest)
            
            # Save best checkpoint
            test_map = results[2]
            if test_map > best_map:
                best_map = test_map 
            if best_map == test_map:
               torch.save(chkpt, best)

            # Save backup every 20 epochs (optional)
#             if epoch > 0 and epoch % 20 == 0:
#                 torch.save(chkpt, weights + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt

################ TRAINING ########
# change data file to train_medAL/train_random and test_random
delta_epochs=35
training_sizes = [0,30,50,100,350,500,1100,1600,2100,2600] # 
for j in range(5):
    if not opt.test_only:
        # training schedule
        for i in range(len(training_sizes)-1):
            epochs=5
            train(cfg='cfg/yolov3-spp.cfg',
                   data_cfg='data/truck_data.data',
                   img_size=416,
                   resume=False, 
                   epochs=epochs,
                   batch_size=16,
                   accumulate=4,
                   multi_scale=False,
                   freeze_backbone=False,
                   num_workers=4,
                   transfer=True,
                   first_img=0,
                   last_img = training_sizes[i+1]
                   )
            epochs+=delta_epochs
            train(cfg='cfg/yolov3-spp.cfg',
                   data_cfg='data/truck_data.data',
                   img_size=416,
                   resume=True, 
                   epochs=epochs,
                   batch_size=16,
                   accumulate=4,
                   multi_scale=False,
                   freeze_backbone=False,
                   num_workers=4,
                   transfer=False,
                   first_img=0,
                   last_img = training_sizes[i+1]
                   )
            shutil.move("weights/best.pt", "weights/{0}_{1}_e{2}.pt".format(0,training_sizes[i+1], epochs))

    ############################# Testing
    epochs=delta_epochs+5
    f = open("incremental_testing_results.txt","a")
    f.write("mp mr map mf1 loss #targets #predicted #correct\n")
    # testing schedule
    for i in range(len(training_sizes)):
        if i != 0:
            with torch.no_grad():
                mp, mr, map, mf1, loss, Y, y, num_correct= test(
                        cfg='cfg/yolov3-spp.cfg',
                        data_cfg='data/truck_data.data',
                        weights="weights/{0}_{1}_e{2}.pt".format(training_sizes[0],training_sizes[i],epochs),
                        batch_size=16,
                        img_size=416,
                        iou_thres=0.5,
                        conf_thres=0.8, # to match Rui's paper
                        nms_thres=0.5,
                        save_json=False,
                        model=None
                       )
            f.write("{0},{1},{2},{3},{4},{5},{6},{7}\n".format(mp, mr, map, mf1, loss, Y, y, num_correct))
    #         epochs+=delta_epochs
    f.close()