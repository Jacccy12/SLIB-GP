Download relevant datasets: CelebA, MNIST.
CelebA
MNIST
python prepare_data.py

The directory of datasets is organized as follows:
```
./attack_dataset
├── MNIST 
│   ├── *.txt 
│   └── Img
│       └── *.png
└── CelebA                            
    ├── *.txt 
    └── Img
        └── *.png
```

# Privacy enhancement with SLIB-GP
    # dataset:celeba, mnist, cifar; 
    # measure:COCO, HSIC; MID ,REG
    # balancing hyper-parameters: tune them in train_mutual+DP.py
    ```
- For KED-MI, if you trained a defense model yourself, you have to train an attack model (generative model) specific to this defense model.
    ```
    # put your hyper-parameters in DMI/k+1_gan_MI.py first
    python k+1_gan_MI --dataset=celeba --defense=MI
    ```
  - For GMI   
  ```
    # python train_gan 
    python attack_MI --dataset=celeba --defense=MI


# Defending against MI attacks 
Here, we only provide the weights file of the well-trained defense models that achieve the best trade-off between model robustness and utility, which are highlighted in the experimental results.
- GMI
    - Weights file (defense model / eval model / GAN) :
        - Place pretrained VGG16in `SLIB-GP/target_model/`
        - Place defense model in `SLIB-GP/target_model/celeba/`

        - Place evaluation classifer in `GMI/eval_model/`
        - Place GAN in `GMI/result/models_celeba_gan/`

    - Launch attack
        ```
        # balancing hyper-parameters: (0.05, 0.5)
        python attack.py --dataset=celeba --defense=SLIB-GP
        ```
    - Calculate FID
        ```
        # sample real images from training set
        cd attack_res/celeba/pytorch-fid && python private_domain.py 
        # calculate FID between fake and real images
        python fid_score.py ../celeba/trainset/ ../celeba/MI/all/
        ```
        
- KED-MI
    - Weights file (defense model / eval model / GAN) :
        - Place defense model in `SLIB-GP/target_model/`
        - Place evaluation classifer in `DMI/eval_model/`
        - Dataset config in `DMI/config/`
        - - Place evaluation classifer in `GMI/eval_ckp/`
        - Place improved GAN for celeba in `DMI/improvedGAN/celeba/MI/`
        
    - Launch attack
        ```
        # balancing hyper-parameters: (0.05, 0.5)
        python recovery.py --dataset=celeba --defense=MI
        # balancing hyper-parameters: (5, 10)
        python recovery.py --dataset=mnist --defense=MI
        ```
    - Calculate FID
        ```
        # celeba
        cd attack_res/celeba/pytorch-fid && python private_domain.py 
        python fid_score.py ../celeba/trainset/ ../celeba/MI/all/ --dataset=celeba
        # mnist
        cd attack_res/mnist/pytorch-fid && python private_domain.py 
        python fid_score.py ../mnist/trainset/ ../mnist/MI/all/ --dataset=mnist
        ```

