import argparse
import subprocess


from pathlib import Path

from source_codes import generate

CUR_PATH = Path.cwd()
Q_IMG_DIR = Path(CUR_PATH/'q_img')
Q_VID_DIR = Path(CUR_PATH/'q_vid')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="큐에 삽입할 파일")
    parser.add_argument("--type", help="확장자 (img/vid)")
    args = parser.parse_args()
    
    FILE_DIR = args.file
    FILE_NAME = Path(FILE_DIR).stem + '.png'
    FILE_TYPE = args.type
    
    start_dir = Path(CUR_PATH.parent/FILE_DIR)
    end_dir = Path(CUR_PATH/f'q_{FILE_TYPE}'/FILE_NAME)
    
    move_cmd = f"mv {start_dir} {end_dir}"
    print(move_cmd)
