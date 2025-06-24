from .ssvep_classifier import Classifier

def get_ssvep_model():
    model = Classifier(
        n_electrodes=3,
        dropout=0.26211635308091535,
        out_dim=4,
        kernLength=256,
        F1 = 32,
        D = 3,
        F2 = 96,
        hidden_dim=256,
        layer_dim=3,
    )
    
    return model
    
def get_mi_model():
    model = Classifier(
        n_electrodes=3,
        dropout=0.26211635308091535,
        out_dim=2,
        kernLength=256,
        F1 = 32,
        D = 3,
        F2 = 32,
        hidden_dim=256,
        layer_dim=3,
    )
    
    return model