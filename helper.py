def load_dataset():
    # isi baris
    raw = []

    # buka file
    f = open('iris.csv')

    # baca per baris
    line = f.readline()

    # selama baris di file masih ada
    while line:
        # isi baris masukkan ke variabel raw
        raw.append(line)

        # pindah baris selanjutnya
        line = f.readline()


    # Hapus baris pertama
    raw = raw[1:]

    # Hapus karakter escape, Pisah berdasarkan koma
    raw = [r.strip().split(',') for r in raw]

    # print(raw[:5])

    # List untuk menampung nilai atribut
    # tiap data dan kelasnya
    data = []
    label = []

    # Perulangan untuk tiap data di dalam variabel
    # raw
    for r in raw:
        # Ambil nilai atribut
        data.append([float(val) for val in r[:4]])

        # Ambil class dari data
        label.append(r[4])

    # Ubah class menjadi nilai numerik
    # 0 = setosa, 2 = virginia, 1 = versicolor
    set_label = list(set(label))
    label = [set_label.index(l) for l in label]

    # Return data, class, dan set dari class
    return data, label, set_label
