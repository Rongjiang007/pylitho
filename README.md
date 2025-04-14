# pylitho
Lithofacies prediction from well log data based on Deep learning

If you want to use a pre-trained model for training, use the command:

predict.py -m filemodel -d datafile

filemodel can be RESNET or CNN to use different models for prediction.
datafile is the data to be predicted, in npy format.

For example in our code:   python3 predict.py -m RESNET -d DATA_lianghe_mulclas.npy

datafile has following numpy matrix formart with 6 columns:
Well name, Depth, NR, GG, GR, labels

Output will be save as in out dir.

If you want to train a new model, use the trainCNN.ipynb, trainRESNET.ipynb and trainRF.ipynb.

## Citation
Shi, Y., Liao, J., Gan, L., & Tang, R. (2024). Lithofacies prediction from well log data based on deep learning: a case study from Southern Sichuan, China. Applied Sciences, 14(18), 8195.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
