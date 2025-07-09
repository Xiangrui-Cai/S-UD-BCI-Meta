import cnn_model

# %%
def getModel(model_name):
    if (model_name == 'CNNNet'):
        model = cnn_model.UDCNN_BiTCN(
            # Dataset parameters
            n_classes=2,
            in_chans=32,
            in_samples=1500,
            eegn_F1=64,
            eegn_D=2,
            eegn_kernelSize=64,
            eegn_dropout=0.3,
        )
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))

    return model

model = getModel('CNNNet')
model.load_weights(r".\model.h5")


def test(X_test):
    # 做预测
    y_pred = model.predict(X_test).argmax(axis=-1)
    print(y_pred)
    return y_pred

def run(X_test):
    result = test(X_test)
    return result


