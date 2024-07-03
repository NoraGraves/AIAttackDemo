# Adapted from https://pyimagesearch.com/2020/10/26/targeted-adversarial-attacks-with-keras-and-tensorflow/

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import numpy as np
import cv2
import AIAttackDemo.image_processing as image_processing
import AIAttackDemo.neural_net as neural_net

EPS = 2 / 255.0
LR = 5e-3

# clip the values of the tensor to a given range and return it
def clip_eps(tensor, eps):
    return tf.clip_by_value(tensor, clip_value_min=-eps,
        clip_value_max=eps)

# Use gradient method and return targeted delta (noise vector)
def generate_targeted_adversaries(model, baseImage, delta, classIdx,
    target, steps, learning_rate, checkin=False):
    # initialize optimizer and loss function
    optimizer = Adam(learning_rate=learning_rate)
    sccLoss = SparseCategoricalCrossentropy()

    # iterate over the number of steps
    for step in range(0, steps):
        # record our gradients
        with tf.GradientTape() as tape:
            # explicitly indicate that our perturbation vector should
            # be tracked for gradient updates
            tape.watch(delta)
            
            # add our perturbation vector to the base image and
            # preprocess the resulting image
            adversary = preprocess_input(baseImage + delta)
            
            # run this newly constructed image tensor through our
            # model and calculate the loss with respect to the
            # both the *original* class label and the *target*
            # class label
            predictions = model(adversary, training=False)
            
            # find current prediction and change classIdx if necessary
            # or end early if goal has been reached
            if checkin:
                classIdx = int(np.argmax(predictions))
                if (steps % 50 == 0):
                    print(f'[CHECK-IN] Current prediction is {neural_net.index2name(classIdx)}')
                if classIdx == target:
                    print('Goal reached. Ending early.')
                    return delta

            originalLoss = -sccLoss(tf.convert_to_tensor([classIdx]),
                predictions)
            targetLoss = sccLoss(tf.convert_to_tensor([target]),
                predictions)
            totalLoss = originalLoss + targetLoss
            
            # check to see if we are logging the loss value, and if
            # so, display it to our terminal
            if step % 20 == 0:
                print("step: {}, loss: {}...".format(step,
                    totalLoss.numpy()))
                        
        # calculate the gradients of loss with respect to the
        # perturbation vector
        gradients = tape.gradient(totalLoss, delta)
        
        # update the weights, clip the perturbation vector, and
        # update its value
        optimizer.apply_gradients([(gradients, delta)])
        delta.assign_add(clip_eps(delta, eps=EPS))
    # return the perturbation vector
    return delta

# Runs a targeted attack on the cropped image and returns the
# adversarial image
def targeted_attack(model, image, target_name, steps=400, eps=EPS, learning_rate=LR):
    target=neural_net.name2index(target_name)
    # Turn image into array
    image = image_processing.preprocess_image(image)

    # create a tensor based off the input image and initialize the
    # perturbation vector (we will update this vector via training)
    baseImage = tf.constant(image, dtype=tf.float32)
    delta = tf.Variable(tf.zeros_like(baseImage), trainable=True)

    print("[INFO] finding original classification...")
    # Find current prediction
    orig_class = neural_net.predict_index(image, model)

    print("[INFO] running the attack...")
    # Run the attack
    deltaUpdated = generate_targeted_adversaries(model=model, baseImage=baseImage, delta=delta,
    classIdx=orig_class, target=target, steps=steps, learning_rate=learning_rate)

    # create the adversarial example, swap color channels, and save the
    # output image to disk
    print("[INFO] creating targeted adversarial example...")
    adverImage = (baseImage + deltaUpdated).numpy().squeeze()
    adverImage = np.clip(adverImage, 0, 255).astype("uint8")
    adverImage = cv2.cvtColor(adverImage, cv2.COLOR_RGB2BGR)
    return adverImage

# Runs a targeted attack on the cropped image and returns the
# adversarial image
def targeted_attack_checkin(model, image, target_name, steps=400, eps=EPS, learning_rate=LR):
    target=neural_net.name2index(target_name)
    # Turn image into array
    image = image_processing.preprocess_image(image)

    # create a tensor based off the input image and initialize the
    # perturbation vector (we will update this vector via training)
    baseImage = tf.constant(image, dtype=tf.float32)
    delta = tf.Variable(tf.zeros_like(baseImage), trainable=True)

    print("[INFO] finding original classification...")
    # Find current prediction
    orig_class = neural_net.predict_index(image, model)

    print("[INFO] running the attack...")
    # Run the attack
    deltaUpdated = generate_targeted_adversaries(model=model, baseImage=baseImage, delta=delta,
    classIdx=orig_class, target=target, steps=steps, learning_rate=learning_rate, checkin=True)

    # create the adversarial example, swap color channels, and save the
    # output image to disk
    print("[INFO] creating targeted adversarial example...")
    adverImage = (baseImage + deltaUpdated).numpy().squeeze()
    adverImage = np.clip(adverImage, 0, 255).astype("uint8")
    adverImage = cv2.cvtColor(adverImage, cv2.COLOR_RGB2BGR)
    return adverImage
