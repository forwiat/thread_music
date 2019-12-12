class hparams:
    def __init__(self):
        self.MULTI_PROCESS = False # Boolean. Do multi processing or not in pre-processing.
        self.CPU_RATE = 0.5 # A Float Number. Cpu rate if multi processing in pre-processing.
        self.CSV_PATH = '' # String. File path, content include wave path and labels. Format: Path \t label_0|label_1|...
        self.SEGMENT = 1000 # An integer. Frame nums.
        self.LABEL_PATH = '' # String. Including total labels.
        self.TF_DIR = '' # String. Tfrecord files directory.
        self.LABEL_DIC = self.__GET_LABEL_DIC__()
        self.LABEL_SIZE = len(self.LABEL_DIC)
        self.SR = 16000
        self.N_FFT = 1024
        self.HOP_LENGTH = int(0.0125 * self.SR)
        self.WIN_LENGTH = int(0.05 * self.SR)
        self.REF_DB = 20
        self.MAX_DB = 100
        self.NUM_EPOCHS = 30
        self.FEATURE_SIZE = (self.N_FFT/2 + 1)
        self.BATCH_SIZE = 64
        self.UNITS = 768
        self.LR = 0.001
        self.DECAY_RATE = 0.5
        self.DECAY_STEPS = 2000

    def __GET_LABEL_DIC__(self):
        import codecs
        lines = codecs.open(self.LABEL_PATH, 'r').readlines()
        dic = {}
        cnt = 0
        for line in lines:
            label = line.strip()
            dic[label] = cnt
            cnt += 1
        return dic
