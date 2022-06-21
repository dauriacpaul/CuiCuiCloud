# Load dependencies
import os, sys, json, re  # , wikipedia
import pandas as pd
from timeit import default_timer as timer
from datetime import datetime
from urllib.request import urlopen
from bs4 import BeautifulSoup

# CodeCarbon
# from codecarbon import EmissionsTracker

print('Now is %s\n' % datetime.now())

# --------------------------------------------------------------------------------------------
# To edit
cfg_new = {}
cfg_new['WORKING_PATH'] = r'C:\Users\SP0043BF\PycharmProjects\CuiCui\birdnet'
cfg_new['CODES_FILE'] = os.path.join(cfg_new['WORKING_PATH'], 'BirdNET', 'eBird_taxonomy_codes_2021E.json')
cfg_new['LABELS_FILE'] = os.path.join(cfg_new['WORKING_PATH'], 'BirdNET', 'BirdNET_GLOBAL_2K_V2.1_Labels.txt')
cfg_new['LABELS_TRANS_FILE'] = os.path.join(cfg_new['WORKING_PATH'], 'BirdNET', 'ebird.json')
cfg_new['MODEL_FILE'] = os.path.join(cfg_new['WORKING_PATH'], 'BirdNET', 'BirdNET_GLOBAL_2K_V2.1_Model_FP32.tflite')
# cfg_new['MDATA_MODEL_FILE'] = os.path.join(cfg_new['WORKING_PATH'], 'BirdNET', 'BirdNET_GLOBAL_2K_V2.1_MData_Model_FP32.tflite')
cfg_new['SPECIES_LIST_FILE'] = cfg_new['LABELS_FILE']
# cfg_new['SPECIES_LIST_FILE'] = os.path.join(cfg_new['WORKING_PATH'], '01_Input', 'species_list.txt')
cfg_new['INPUT_PATH'] = os.path.join(cfg_new['WORKING_PATH'], '01_Input')
cfg_new['INPUT_FILES'] = [
    # os.path.join(cfg_new['INPUT_PATH'], 'soundscape.wav'),
    # os.path.join(cfg_new['INPUT_PATH'], 'XC509128 - Chouette-pêcheuse de Bouvier - Scotopelia bouvieri.mp3'),
    os.path.join(cfg_new['INPUT_PATH'], 'XC509128-Chouette-pêcheuse-de-Bouvier-Scotopelia-bouvieri.wav'),
    os.path.join(cfg_new['INPUT_PATH'], 'Fauvette à tête noire 2022-06-09 07_56_32 à Toulouse.wav'),
    #   os.path.join(cfg_new['INPUT_PATH'], 'Fauvette-à-tête-noire-2022-06-09-07_56_32-à-Toulouse.mp3'),
    os.path.join(cfg_new['INPUT_PATH'], 'Chardonneret élégant 2022-06-12 18_37_18 à Toulouse.wav'),
]
cfg_new['OUTPUT_PATH'] = os.path.join(cfg_new['WORKING_PATH'], '02_Output')
cfg_new['OUTPUT_DO'] = True
cfg_new['ERROR_LOG_FILE'] = os.path.join(cfg_new['OUTPUT_PATH'],
                                         '%s__Error_log.txt' % datetime.now().strftime('%Y%m%d'))
# Audio settings
cfg_new['SIG_LENGTH'] = 3.0  # 3-second chunks
cfg_new[
    'SAMPLE_RATE'] = 48000  # Sample rate of 48kHz, so the model input size is (batch size, 48000 kHz * 3 seconds) = (1, 144000)
cfg_new['SIG_OVERLAP'] = 0  # 0 = no overlap; overlap between consecutive chunks <3.0
cfg_new['SIG_MINLEN'] = 2.0  # Minimum length of audio chunk for prediction
# Inference settings
cfg_new['MIN_CONFIDENCE'] = .1
# cfg_new['MIN_CONFIDENCE'] = .5
cfg_new['SENSITIVITY'] = 1
cfg_new['OVERLAP_SEC'] = 0
# cfg_new['RESULT_TYPE'] = 'csv'  # table / audacity / csv
cfg_new['THREADS'] = 4
# --------------------------------------------------------------------------------------------

# Load BirdNET
sys.path.append(os.path.join(cfg_new['WORKING_PATH'], 'BirdNET'))
from birdnet.BirdNET import analyze_NCH  # Analyze audio files with BirdNET
from birdnet.BirdNET import config as cfg  # Configuration file (model filename, audio settings, settings,...)


def get_photo_url(name):
    html = urlopen('https://en.wikipedia.org/wiki/' + name.replace(' ', '_'))
    bs = BeautifulSoup(html, 'html.parser')
    image = bs.find_all('img', {'src': re.compile('.jpg')})[0]['src'].split('.jpg')[0] + '.jpg'
    return image.replace('//upload.wikimedia.org/wikipedia/commons/thumb/',
                         'https://upload.wikimedia.org/wikipedia/commons/')


# def get_photo_url_fr(name):
#   wikipedia.set_lang('fr')
#   search_results = wikipedia.search(name.replace('Ã©', 'é'))
#   return [x for x in wikipedia.page(search_results[0]).images if x.endswith('.jpg')][0]

if __name__ == '__main__':

    # Update cfg
    cfg.CODES_FILE = cfg_new['CODES_FILE']
    cfg.LABELS_FILE = cfg_new['LABELS_FILE']
    cfg.MODEL_PATH = cfg_new['MODEL_FILE']
    # cfg.MDATA_MODEL_PATH = cfg_new['MDATA_MODEL_FILE']
    cfg.INPUT_PATH = cfg_new['INPUT_PATH']
    cfg.OUTPUT_PATH = cfg_new['OUTPUT_PATH']
    cfg.ERROR_LOG_FILE = cfg_new['ERROR_LOG_FILE']
    cfg.SAMPLE_RATE = cfg_new['SAMPLE_RATE']
    cfg.SIG_LENGTH = cfg_new['SIG_LENGTH']
    cfg.SIG_OVERLAP = cfg_new['SIG_OVERLAP']
    cfg.SIG_MINLEN = cfg_new['SIG_MINLEN']
    cfg.MIN_CONFIDENCE = max(0.01, min(0.99, float(cfg_new['MIN_CONFIDENCE'])))
    cfg.SIGMOID_SENSITIVITY = max(0.5, min(1.0 - (float(cfg_new['SENSITIVITY']) - 1.0), 1.5))
    cfg.SIG_OVERLAP = max(0.0, min(2.9, float(cfg_new['OVERLAP_SEC'])))
    # cfg.RESULT_TYPE = cfg_new['RESULT_TYPE'].lower()
    cfg.TFLITE_THREADS = int(cfg_new['THREADS'])
    if os.path.exists(cfg_new['SPECIES_LIST_FILE']):
        cfg.SPECIES_LIST = analyze_NCH.loadSpeciesList(cfg_new['SPECIES_LIST_FILE'])
    else:
        cfg.SPECIES_LIST = pd.DataFrame()
    if os.path.exists(cfg_new['LABELS_TRANS_FILE']):
        labels_fr = pd.DataFrame(json.loads(open(cfg_new['LABELS_TRANS_FILE'], 'r').read()))
    else:
        labels_fr = pd.DataFrame()

    # Load eBird codes, labels
    cfg.CODES = analyze_NCH.loadCodes()
    cfg.LABELS = analyze_NCH.loadLabels(cfg.LABELS_FILE)
    # cfg.TRANSLATED_LABELS = analyze_NCH.loadLabels(cfg.LABELS_FILE)

    print('Running BirdNET on %d file(s) with min confidence of %d%%...\n' % (
        len(cfg_new['INPUT_FILES']), cfg.MIN_CONFIDENCE * 1e2))
    ttime = timer()

    # Process files
    results = {}
    col_names = ['ts_start', 'ts_stop', 'sciName', 'comName_en', 'confidence']
    for i, f in enumerate(cfg_new['INPUT_FILES']):
        p_start = timer()
        print('%2d. === Processing file "%s" (%.2f Mo)... %s' % (
            i + 1, os.path.basename(f), os.path.getsize(f) / (1024 ** 2), '=' * (90 - 10 - len(os.path.basename(f)))))
        results[os.path.basename(f)] = pd.DataFrame(analyze_NCH.predictFile((f, cfg.getConfig())),
                                                    columns=col_names).merge(labels_fr, on='sciName', how='left')
        ts = [0, 0]
        for row in results[os.path.basename(f)].itertuples():
            st = '\t%s (aka %s) %sconf. %.2f%%' % (
                row.comName, row.sciName, ' ' * (50 - len(row.comName) - len(row.sciName)), row.confidence * 1e2)
            if ts[0] < row.ts_start or ts[1] < row.ts_stop:
                ts = (row.ts_start, row.ts_stop)
                print('    [%.2f > %.2f sec]   %s' % (ts[0], ts[1], st))
            else:
                print('                        %s' % st)
        print('    === ... Done in %.6f seconds! %s' % (timer() - p_start, '=' * 82))
        print('    Adding photo url for %d bird name(s)...' % results[os.path.basename(f)].shape[0])
        results[os.path.basename(f)]['photo_url'] = results[os.path.basename(f)]['comName_en'].apply(get_photo_url)
        # results[os.path.basename(f)]['photo_url'] = results[os.path.basename(f)]['comName'].apply(get_photo_url_fr)
        if 'OUTPUT_DO' in cfg_new.keys() and cfg_new['OUTPUT_DO']:
            fname = os.path.splitext(os.path.basename(f))[0] + '.xlsx'
            print('    Writing "%s"...' % fname)
            results[os.path.basename(f)].to_csv(
                os.path.join(cfg_new['OUTPUT_PATH'], os.path.splitext(os.path.basename(f))[0] + '.csv'),
                encoding='iso-8859-1', sep=';')
        print()

    print('Elapsed time is %.4f seconds.' % (timer() - ttime))

print('\nNow is %s' % datetime.now())
