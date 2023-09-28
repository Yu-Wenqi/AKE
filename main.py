# -*- coding: utf-8 -*-


import logging
from config import *
from Bilstm_crf import blcrf
from bilstm import bl
from attbilstm import att_bl
from attbilstm_crf import att_blcrf
from BERT import BERT
from macBERT import macBERT
from RoBERTa import RoBERta


if __name__ == '__main__':

   # answerP = []
   # answerR = []
   # answerF = []

   for i in range(1):

       # 将结果打印到日志中
       logging.basicConfig(level=logging.DEBUG,
                           format='%(asctime)s - %(lineno)d - %(levelname)s: %(message)s',
                           datefmt='%m-%d %H:%M',
                           filename=logfile,  # 文件内容
                           filemode='a')  # w覆盖写入，a继续写入

       #selfattention_BiLSTM_CRF
       # best_P, best_R, best_F= satt_blcrfpro(train_path, test_path, vocab_path)

       # bl
       # epoch3,epoch5,epoch10,best_P3, best_R3, best_F3, best_P5, best_R5, best_F5 , best_P10, best_R10, best_F10= bl(train_path, test_path, vocab_path)
       # best_P, best_R, best_F, best_10F, best_15F = bl(train_path, test_path, vocab_path)

       #blcrf
       # epoch3,epoch5,epoch10,best_P3, best_R3, best_F3, best_P5, best_R5, best_F5 , best_P10, best_R10, best_F10 = blcrf(train_path, test_path, vocab_path)

       # att—bl
       # epoch3,epoch5,epoch10,best_P3, best_R3, best_F3, best_P5, best_R5, best_F5 , best_P10, best_R10, best_F10 = att_bl(train_path, test_path, vocab_path)

       # att_bilstmcrf
       epoch3,epoch5,epoch10,best_P3, best_R3, best_F3, best_P5, best_R5, best_F5 , best_P10, best_R10, best_F10 = att_blcrf(train_path, test_path, vocab_path)

       # BERT
       # epoch3, epoch5, epoch10, best_P3, best_R3, best_F3, best_P5, best_R5, best_F5, best_P10, best_R10, best_F10 = BERT(train_path, test_path, vocab_path)

       # macBERT
       # epoch3, epoch5, epoch10, best_P3, best_R3, best_F3, best_P5, best_R5, best_F5, best_P10, best_R10, best_F10 = macBERT(train_path, test_path, vocab_path)

       # answerP.append(best_P)
       # answerR.append(best_R)
       # answerF.append(best_F)

       logging.info("The feature ：" + feature)

       print("The best P@3 of first ",epochs ,"epoch is", best_P3)
       logging.info("The best P@3 first"+str(epochs)+"epoch is:" + str(best_P3))
       print("The best R@3 of first ",epochs ," epoch is", best_R3)
       logging.info("The best R@3 first "+str(epochs)+" epoch is:" + str(best_R3))
       print("The best F@3 of first ",epochs ," epoch is", best_F3)
       logging.info("The best F@3 first "+str(epochs)+" epoch is:" + str(best_F3))
       print(str(best_P3)+" "+str(best_R3)+" "+str(best_F3))
       logging.info(str(epoch3)+":"+str(best_P3)+" "+str(best_R3)+" "+str(best_F3))

       print("The best P(5) of first ",epochs ," epoch is", best_P5)
       logging.info("The best P(5) first "+str(epochs)+" epoch is:" + str(best_P5))
       print("The best R(5) of first ",epochs ," epoch is", best_R5)
       logging.info("The best R(5) first "+str(epochs)+" epoch is:" + str(best_R5))
       print("The best F(5) of first ",epochs ," epoch is", best_F5)
       logging.info("The best F(5) first "+str(epochs)+" epoch is:" + str(best_F5))
       print(str(best_P5)+" "+str(best_R5)+" "+str(best_F5))
       logging.info(str(epoch5)+":"+str(best_P5)+" "+str(best_R5)+" "+str(best_F5))

       print("The best P(10) of first ",epochs ," epoch is", best_P10)
       logging.info("The best P(10) first "+str(epochs)+" epoch is:" + str(best_P10))
       print("The best R(10) of first ",epochs ," epoch is", best_R10)
       logging.info("The best R(10) first "+str(epochs)+" epoch is:" + str(best_R10))
       print("The best F(10) of first ",epochs ," epoch is", best_F10)
       logging.info("The best F(10) first "+str(epochs)+" epoch is:" + str(best_F10))
       print(str(best_P10)+" "+str(best_R10)+" "+str(best_F10))
       logging.info(str(epoch10)+":"+str(best_P10)+" "+str(best_R10)+" "+str(best_F10))

       # print("The best F first 10 epoch is", best_10F)
       # logging.info("The best F first 10 epoch is:" + str(best_10F))
       # print("The best F first 15 epoch is", best_15F)
       # logging.info("The best F first 15 epoch is:" + str(best_15F))
       # print("The best P of first 20 epoch is", best_P)
       # logging.info("The best P first 20 epoch is:" + str(best_P))
       # print("The best R of first 20 epoch is", best_R)
       # logging.info("The best R first 20 epoch is:" + str(best_R))
       # print("The best F of first 20 epoch is", best_F)
       # logging.info("The best F first 20 epoch is:" + str(best_F))

       logging.info("The learning rate is:" + str(lr))
       logging.info("The dropout rate is:" + str(dropout_value))
       logging.info("The weight decay is:" + str(weight_decay))
       # logging.info("The 5 fold is end.")

