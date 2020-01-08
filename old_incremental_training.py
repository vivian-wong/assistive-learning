import argparse
import time

import torch.distributed as dist
from torch.utils.data import DataLoader

import test  # Import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
import shutil


def train(
        cfg,
        data_cfg,
        img_size=416,
        resume=False,
        epochs=273,  # 500200 batches at bs 64, dataset length 117263
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

        # Calculate mAP 
        # removed cuz this took too long 

        # with torch.no_grad():
        #    results = test.test(cfg, data_cfg, batch_size=batch_size, img_size=img_size, model=model)

        # Write epoch results
        #with open('results.txt', 'a') as file:
        #    file.write(s + '%11.3g' * 5 % results + '\n')  # P, R, mAP, F1, test_loss

        # Update best loss
        #test_loss = results[4]
        #if test_loss < best_loss:
        #    best_loss = test_loss

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
            #if best_loss == test_loss:
            #    torch.save(chkpt, best)

            # Save backup every 20 epochs (optional)
            if epoch > 0 and epoch % 20 == 0:
                torch.save(chkpt, weights + 'backup%g.pt' % epoch)

            # Delete checkpoint
            # del chkpt

training_sizes = [0,30,35,45,70,100,200,400,800,1600,3501] # do only 0-800 only. evaluate on 1600-3501
epochs=0
# training schedule
for i in range(len(training_sizes)-2):
    if i == 0:
        train(cfg='cfg/yolov3-spp.cfg',
               data_cfg='data/truck_data.data',
               img_size=416,
               resume=False, 
               epochs=1,
               batch_size=16,
               accumulate=1,
               multi_scale=False,
               freeze_backbone=False,
               num_workers=4,
               transfer=True,
               first_img=training_sizes[i],
               last_img = training_sizes[i+1]
               )
        train(cfg='cfg/yolov3-spp.cfg',
               data_cfg='data/truck_data.data',
               img_size=416,
               resume=True, 
               epochs=99,
               batch_size=16,
               accumulate=1,
               multi_scale=False,
               freeze_backbone=False,
               num_workers=4,
               transfer=False,
               first_img=training_sizes[i],
               last_img = training_sizes[i+1]
               )
        shutil.copy("weights/latest.pt", "weights/{0}_{1}_e100.pt".format(training_sizes[i],training_sizes[i+1]))
        epochs=100
    else:
        epochs=epochs+40
        train(cfg='cfg/yolov3-spp.cfg',
                data_cfg='data/truck_data.data',
                img_size=416,
                resume=True, 
                epochs=epochs,
                batch_size=16,
                accumulate=1,
                multi_scale=False,
                freeze_backbone=False,
                num_workers=4,
                transfer=False,
                first_img=training_sizes[i],
                last_img = training_sizes[i+1]
                )
        shutil.copy("weights/latest.pt", "weights/{0}_{1}_e{2}.pt".format(training_sizes[i],training_sizes[i+1],epochs))
