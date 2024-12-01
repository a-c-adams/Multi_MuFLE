import numpy as np
from scipy.io import loadmat
import glob
import os

# File loading function
def FileLoad(FolderPath, FilePath, FileType, WaveFilePath, TimeFilePath):
    """ A file to load the histogram, wavelength and time data into dictionaries
    :param FolderPath: Path to the main folder the data is stored in
    :param FilePath:Path from the main data folder into the data file to be loaded
    :param FileType: File name to be loaded
    :param WaveFilePath: Path of the lambda map file from the folder the data is stored in
    :param TimeFilePath: Path to the TDCres file from the folder the data is stored in
    :return: Two dictionaries, FileDict containing the histogram, TechDict containing the wavelength and time information
    """
    # Create empty dictionaries
    FileDict = {}
    TechDict = {}

    # Create empty lists to store the data prior to putting them as keys into the dictionaries
    FileList = []
    AllHistData = []

    # DataPath create
    data_path = os.path.join(FolderPath, FilePath)
    DataPath = os.chdir(data_path)

    if DataPath == None:
        DataFile = data_path + FileType
    else:
        DataFile = DataPath + FileType

    # Loading each individual tissue/reference file, adding the files to the FileList list
    for file in glob.glob(DataFile):
        if file == DataFile:
            FileList.append(loadmat(DataFile, struct_as_record=False))
        else:
            FileList.append(loadmat(file, struct_as_record=False))

    for file in FileList:
        ### for the old data it is called histData but from the new data it is called histAll
        ### there is also no settings...
        # HistData = file['histData']
        HistData = file['HistData']
        AllHistData.append(HistData)

        FileDict['HistData'] = AllHistData

    # load the wavelength file and extract the wavelength
    WavePath = FolderPath + WaveFilePath

    WaveFile = loadmat(WavePath, struct_as_record=False)
    WaveOrg = WaveFile['lambdaMap'][0]

    # Load the time file, extracting the different TDCres and histMode from the files
    TimePath = FolderPath + TimeFilePath

    TimeFile = loadmat(TimePath, struct_as_record=False)
    TDCres = TimeFile['TDCres'][0]

    TechDict['TDC'] = TDCres
    TechDict['WaveOrg'] = WaveOrg

    return FileDict, TechDict

def TCSPCtime(TechDict):
    """ File to create the TCSPC time histogram
    :param TechDict: The dictionary containing the TDC
    :return: The time histogram
    """
    TimeBin = np.reshape(np.tile(np.arange(0, 1200, 1), 512), (512, 1200))
    TCSPCTDC = np.reshape(np.tile(TechDict['TDC'], 1200), (512, 1200), order='F')
    return TimeBin * TCSPCTDC