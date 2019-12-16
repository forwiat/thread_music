class hparams:
    def __init__(self):
        #################################################################################
        #                                                                               #
        #                            Preprocess Hyperparams                             #
        #                                                                               #
        #################################################################################

        # ------------------------ Paths And Directory ------------------------ #
        self.CSV_PATH = './dataset/200_music.csv' # String. File path, content include wave path and labels. Format: Path \t label_0|label_1|...
        self.LABEL_PATH = './dataset/total_labels.csv' # String. Including total labels.
        self.TF_DIR = './dataset/train' # String. Tfrecord files directory.
        self.MULTI_PROCESS = False # Boolean. Do multi processing or not in pre-processing.
        self.CPU_RATE = 0.5 # A Float Number. Cpu rate if multi processing in pre-processing.

        # ------------------------ Wave Hyperparams --------------------------- #
        self.SEGMENT = 1000 # An integer. Frame nums.
        self.SR = 16000
        self.N_FFT = 1024
        self.HOP_LENGTH = int(0.0125 * self.SR)
        self.WIN_LENGTH = int(0.05 * self.SR)
        self.REF_DB = 20
        self.MAX_DB = 100
        #################################################################################
        #                                                                               #
        #                               Train Hyperparams                               #
        #                                                                               #
        #################################################################################

        # ------------------------ Paths And Directory ------------------------ #
        self.LOG_DIR = './logs_200_test'
        self.MODEL_DIR = './model_200_test'

        # ------------------------ Train Params ------------------------------- #
        self.NUM_EPOCHS = 20
        self.BATCH_SIZE = 10
        self.LR = 0.001
        self.DECAY_RATE = 0.5
        self.DECAY_STEPS = 10
        self.GPU_IDS = [0] # A list. If length > 1, then multi gpu in training.
        self.PER_STEPS = 10

        # ------------------------ Network Dimensions ------------------------ #
        self.LABEL_DIC = self.__GET_LABEL_DIC__()
        self.LABEL_SIZE = len(self.LABEL_DIC)
        self.UNITS = 768
        self.FEATURE_SIZE = int(self.N_FFT/2 + 1)


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
