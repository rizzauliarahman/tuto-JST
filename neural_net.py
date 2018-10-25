# Impor library untuk membantu perhitungan
import numpy as np
import random


class NeuralNetwork:
    # Konstruktor class
    # Epsilon = learning rate
    # Reg_lambda = Regularization
    # set_label = ['setosa', 'versicolor', 'virginica']
    def __init__(self, x_train, y_train, set_label, epsilon, reg_lambda):
        # random.seed(0)

        # Ubah data dan kelas dari list menjadi array
        x = np.array(x_train)
        y = np.array(y_train)

        # 40 data sebagai data training
        # 10 data sebagai data testing
        # DARI SETIAP KELAS YG ADA

        # [0, 1, 2, ....., 150]
        idx = [i for i in range(len(x))]
        random.shuffle(idx)

        # Ambil 40 data dan kelas pertama sebagai data training
        self.x_train = x[idx[:40]]
        self.y_train = y[idx[:40]]

        # data_training berbentuk list dengan dimensi 40x4

        # Ambil 10 data dan kelas terakhir sebagai data testing
        self.x_test = x[idx[40:]]
        self.y_test = y[idx[40:]]

        # Cari dimensi input
        # Cari dimensi output = Jumlah kelas yg ada (3)
        self.input_dim = self.x_train.shape[1]
        self.output_dim = len(set_label)

        ##
        self.epsilon = epsilon
        self.reg_lambda = reg_lambda

        self.model = {}

    # Fungsi untuk training JST
    # n_hid = jumlah neuron pada hidden layer
    # n_epochs = batas perulangan training JST
    def build_model(self, n_hid=3, n_epochs=50000):
        # np.random.seed(0)

        # W1 = Random 0-1, sebanyak 4x3
        # 0.2, 0.5, 0.6
        # 0.1, 0.3, 0.6
        # 0.1, 0.4, 0.6
        # 0.1, 0.2, 0.5
        W1 = np.random.randn(self.input_dim, n_hid) / np.sqrt(self.input_dim)

        # b1 = Bias
        # Nilai 0, ukurannya 1x3
        b1 = np.zeros((1, n_hid))

        # W2 = Random 0-1, sebanyak 3x3
        W2 = np.random.randn(n_hid, self.output_dim) / np.sqrt(n_hid)

        # b2 = Bias
        # Nilai 0, ukurannya 1x3
        b2 = np.zeros((1, self.output_dim))

        # Perulangan training JST, sesuai nilai n_epochs
        # Perulangan sebanyak 50000 kali
        for i in range(n_epochs):
            ## FORWARD PASS ##
            # z1 = (nilai atribut pada data x W1) + b1
            z1 = self.x_train.dot(W1) + b1

            # aktivasi = 1 / (1 + exp(-z1))
            a1 = np.tanh(z1)

            # z2 = (nilai aktivasi x W2) + b2
            z2 = a1.dot(W2) + b2

            # probabilitas = exp(z2) / jumlah z2
            # [0.5, 0.2, 0.3]
            probs = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)

            ## BACKWARD PASS ##
            # delta3 = probabilitas tiap kelas untuk suatu data
            delta_3 = probs
            delta_3[range(len(self.x_train)), self.y_train] -= 1

            # transpose a1 (baris -> kolom, kolom -> baris),
            # dikalikan dengan nilai variabel delta3
            dW2 = (np.transpose(a1)).dot(delta_3)

            # jumlahkan semua nilai yg ada pada variabel delta3
            db2 = np.sum(delta_3, axis=0, keepdims=True)

            # delta2 = delta3 dikali W2 yg sudah di transpose
            # dikalikan dengan 1 - a1^2
            delta_2 = delta_3.dot(W2.T) * (1 - np.square(a1))

            # data training di transpose
            # dikalikan dengan delta2
            dW1 = (np.transpose(self.x_train)).dot(delta_2)

            # Jumlah dari nilai yg ada pada variabel delta2
            db1 = np.sum(delta_2, axis=0)

            # Kalikan nilai regularisasi dengan nilai W2 dan W1
            # Jumlahkan ke turunan W2 dan W1
            dW2 += self.reg_lambda * W2
            dW1 += self.reg_lambda * W1

            # Kalikan turunan W1, W2, b1, dan b2 dengan
            # learning rate
            W1 += -self.epsilon * dW1
            W2 += -self.epsilon * dW2
            b1 += -self.epsilon * db1
            b2 += -self.epsilon * db2

            # Simpan nilai Weight dan bias ke dalam variabel model
            self.model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

            # Tiap 1000 perulangan, tampilkan akurasi dan nilai
            # loss pada proses training
            if i % 1000 == 0:
                loss, acc = self.count_loss()
                print('Loss at iteration-%i: %.3f, Accuracy: %.2f%%' % (i, loss, acc))

    # Fungsi menghitung loss dan akurasi training
    def count_loss(self):
        z1 = self.x_train.dot(self.model['W1']) + self.model['b1']
        a1 = np.tanh(z1)
        z2 = a1.dot(self.model['W2']) + self.model['b2']

        probs = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)

        # Kelas hasil prediksi untuk tiap data
        pred = np.argmax(probs, axis=1)

        # Hitung hasil prediksi yang benar (sesuai dengan target)
        correct_pred = [int(pred[i] == self.y_train[i]) for i
                        in range(len(pred))]

        # Hitung akurasi
        acc = (sum(correct_pred) / len(pred)) * 100

        # Softmax Loss
        # loss = jumlah(-log(probabilitas))
        loss = np.sum(-np.log(probs[range(len(self.x_train)), self.y_train]))
        loss += (self.reg_lambda/2) * (np.sum(np.square(self.model['W1'])) +
                                       np.sum(np.square(self.model['W2'])))

        # Return loss dan akurasi
        return 1./len(self.x_train) * loss, acc

    # def predict(self):
    #     z1 = self.x_test.dot(self.model['W1']) + self.model['b1']
    #     a1 = np.tanh(z1)
    #     z2 = a1.dot(self.model['W2']) + self.model['b2']
    #
    #     probs = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)
    #
    #     pred = np.argmax(probs, axis=1)
    #     correct_pred = [int(pred[i] == self.y_test[i]) for i in range(len(pred))]
    #     acc = (sum(correct_pred) / len(pred)) * 100
    #
    #     print('Testing Accuracy : %.2f%%' % acc)
