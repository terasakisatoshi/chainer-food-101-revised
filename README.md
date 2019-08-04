# chainer-food-101-revised

Classify category of food with Chainer

- This is an implementation of food 101 Classification with Chainer
- This repository is revesed version of my repository chainer-food-101. The old repository contains some bugs (Sorry my delay response...) and ugly implementation, so please use the revised repository instead of old one.

- The difference between old one is ...
   - Clean structured implementation
   - Update The latest version Chainer/CuPy
   - Use ChainerCV to transform images for data augmentation.
      - This library is very useful for Chainer users.
   - Provide example that use ResNet50 provided ChainerCV project on training.
   - Multi GPU training is supported
     - `$ python train.py --device 0 1`
   - add notebook to observe dataset

# How to train

## prepare dataset

- Download [Food-101 Data Set: Food-101 -- Mining Discriminative Components with Random Forests](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)

```console
$ mkdir ~/dataset
$ cd ~/dataset
$ wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
$ tar xfvz http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
```

- Then the structure of directory under this repository should be...

```
tree -d ~/dataset/food-101
├── images
│   ├── apple_pie
│   ├── baby_back_ribs
│   ├── baklava
│   ├── beef_carpaccio
│   ├── beef_tartare
│   ├── beet_salad
│   ├── beignets
│   ├── bibimbap
│   ├── bread_pudding
│   ├── breakfast_burrito
│   ├── bruschetta
│   ├── caesar_salad
│   ├── cannoli
│   ├── caprese_salad
│   ├── carrot_cake
│   ├── ceviche
│   ├── cheese_plate
│   ├── cheesecake
│   ├── chicken_curry
│   ├── chicken_quesadilla
│   ├── chicken_wings
│   ├── chocolate_cake
│   ├── chocolate_mousse
│   ├── churros
│   ├── clam_chowder
│   ├── club_sandwich
│   ├── crab_cakes
│   ├── creme_brulee
│   ├── croque_madame
│   ├── cup_cakes
│   ├── deviled_eggs
│   ├── donuts
│   ├── dumplings
│   ├── edamame
│   ├── eggs_benedict
│   ├── escargots
│   ├── falafel
│   ├── filet_mignon
│   ├── fish_and_chips
│   ├── foie_gras
│   ├── french_fries
│   ├── french_onion_soup
│   ├── french_toast
│   ├── fried_calamari
│   ├── fried_rice
│   ├── frozen_yogurt
│   ├── garlic_bread
│   ├── gnocchi
│   ├── greek_salad
│   ├── grilled_cheese_sandwich
│   ├── grilled_salmon
│   ├── guacamole
│   ├── gyoza
│   ├── hamburger
│   ├── hot_and_sour_soup
│   ├── hot_dog
│   ├── huevos_rancheros
│   ├── hummus
│   ├── ice_cream
│   ├── lasagna
│   ├── lobster_bisque
│   ├── lobster_roll_sandwich
│   ├── macaroni_and_cheese
│   ├── macarons
│   ├── miso_soup
│   ├── mussels
│   ├── nachos
│   ├── omelette
│   ├── onion_rings
│   ├── oysters
│   ├── pad_thai
│   ├── paella
│   ├── pancakes
│   ├── panna_cotta
│   ├── peking_duck
│   ├── pho
│   ├── pizza
│   ├── pork_chop
│   ├── poutine
│   ├── prime_rib
│   ├── pulled_pork_sandwich
│   ├── ramen
│   ├── ravioli
│   ├── red_velvet_cake
│   ├── risotto
│   ├── samosa
│   ├── sashimi
│   ├── scallops
│   ├── seaweed_salad
│   ├── shrimp_and_grits
│   ├── spaghetti_bolognese
│   ├── spaghetti_carbonara
│   ├── spring_rolls
│   ├── steak
│   ├── strawberry_shortcake
│   ├── sushi
│   ├── tacos
│   ├── takoyaki
│   ├── tiramisu
│   ├── tuna_tartare
│   └── waffles
└── meta
```

## Prepare Python Modules

For examle

- Python
- NumPy, Matplotlib, OpenCV
- Chainer(6.2.0), CuPy(6.2.0), ChainerCV(0.13.1)
  - Note that the vesion of Chainer and CuPy must be same.

## Start training:

- Just do it:

```
$ python train.py
```

- If you like to train using multi gpu, please add `--device` option to specify which gpu use on training. For example

```
$ python train.py --device 0 1
```

- For more information e.g. how to configure batch size ..., just try:

```
$ python train.py --help
```

O.K.

## Evaluation

- just run:

```
$ python predict.py
```
