import helper
import neural_net as neural
import warnings


def main():
    warnings.filterwarnings("ignore")

    # Load dataset
    data, label, set_label = helper.load_dataset()

    # Konstruktor JST
    nn = neural.NeuralNetwork(data, label, set_label, epsilon=1e-3, reg_lambda=0.01)

    # Training JST
    nn.build_model(n_hid=5, n_epochs=20000)

    # print('\n=======================================================================================================\n')
    #
    # nn.predict()


if __name__ == '__main__':
    main()
