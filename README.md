# ViVit-csgo-imitation
Description
-
Counter-Strike Retake with Large-Scale Behavioural Cloning

Our team inspired by cambridge Tim Pearce work (https://arxiv.org/abs/2104.04258).

We want to use different backbone in cloning and apply it in Retake mode.

Structure
-
Our network choose different backbone compare Tim's. We use ViVit, try to solve sequential promblem.

Our input are 100 frames picture. First network block is CNN, learning the feature of picture. Second is ViVit, solving sequential promblem and making decision.
Third is FC-layer, getting the buttons result.
