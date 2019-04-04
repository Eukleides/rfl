
import data_load as dl

dl.set_seed()
fd = dl.FDDB_dataset()
X_train, X_test, y_train, y_test = fd.generate_training_data(verbose=1)

fd.display_image(X_train[10], y_train[10])