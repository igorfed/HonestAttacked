
__version__ = '0.1'
__author__ = 'Igofed'

import argparse
import ImProc

import os



def argParse():

    Honest = "C:/Users/igofed/Downloads/presentation_attacks/frames"
    Attacked = "C:/Users/igofed/Downloads/presentation_attacks/MyTestAttacked"
    Frame = [122176, 122171, 122173, 122172, 122175, 122174, 121747, 121987, 121957, 121703, 121693, 111111, 222222]
    parser = argparse.ArgumentParser(description="Process Honest and Atacked images")
    output = ImProc.mkDir(dirName="222222")
    parser.add_argument("-Honest", "--Honest",
                        required=False,
                        type=str,
                        default= Honest, help = 'Folder with a honest authentication attempts')

    parser.add_argument("-Attacked", "--Attacked",
                        required=False,
                        type=str,
                        default= Attacked, help = 'Folder with a impostor authentication attempts')

    parser.add_argument("-Output",
                        "--Output",
                        required=False,
                        type=str,
                        default= output)
    parser.add_argument("-Frame",
                        "--Frame",
                        required=False,
                        type=str,
                        default=Frame[12])

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = argParse()

    __mt = ImProc.ImProcess(folder1 = args.Honest,
                            folder2 =  args.Attacked,
                            output = args.Output,
                            Frame = args.Frame)

