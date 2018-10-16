This is an adaption of the repo https://github.com/kenshohara/3D-ResNets-PyTorch, extended to include C3D and random label training

How the dataloaded expects the data:


Assume following directory structure: 

```
/data/
    UCF-101/jpg
    results/
    ucf101_01.json
```

To train a model type in this command into the command line:

`python main.py --root_path /data --video_path UCF-101/jpg --annotation_path ucf101_01.json \
--result_path results --dataset ucf101 --model resnet \
--model_depth 34 --n_classes 101 --batch_size 128 --n_threads 4 --checkpoint 5`

The arguments required by the dataloader are root_path, video_path, annotation_path, the dataset variable decides which dataloader to use (options are ucf101, hmdb51, activitynet and sports1m)
