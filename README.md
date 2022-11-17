
# ANN Challenge team JAN

I rearranged the repository structure, added a `.gitignore` file and split the code into two scripts:

* `script.py` trains the network with the frozen VGG16 model
it is called like this: `python3 script.py model_name.hb5` where `model_name.hb5` is the file where the script will save the first model
* `fine_tuning.py` is used for tuning the previous model. It is called like `python3 script.py model_name.hb5` where `model_name.hb5` is the name of the model it will fine-tune. The final model is saved as `model.hb5`