import os
import pandas as pd
import numpy as np
import csv
import sys
import copy
import argparse
sys.path.append(os.path.abspath("bigdancing_holistic"))
from mvc import  read_graph,min_vertex_cover,read_graph_dc
from tes import greedy_min_vertex_cover,greedy_min_vertex_cover_dc
from tqdm import tqdm
from DC_Rules import DCRule
import time
import tracemalloc
import signal
from datetime import datetime
import re


def handler(signum, frame):
    raise TimeoutError("Time exceeded")

class BigDansing():
    def __init__(self):
        self.wrong = []
        self.rule = []
        self.blocked_list = [[]]
        self.data = []
        self.data_cl = []
        self.input_data = []
        self.dicc = {}
        self.visdic = {}
        self.edgedic = {}
        self.diccop = {}
        self.maypair = [[]]
        self.attr_index = {}
        self.dic = {}
        self.scodic = {}
        self.clean_right = 0
        self.all_clean = 0
        self.clean_right_pre = 0
        self.Rules = []
        self.exps = []
        self.mvc = []
        self.sorts = []
        self.vio = []
        self.cnt = 0
        self.constant_pre = []
        self.contantdic = {}

    def scope(self, sco):
        df = pd.read_csv(dirty_path,  header=0).astype(str).fillna("nan")
        data = np.array(df[sco]).tolist()
        for i in data:
            for j in i:
                j = str(j)
        return data

    def scope1(self, sco):
        # #数据集beers预处理
        # print(sco)
        df = pd.read_csv(clean_path,  header=0).astype(str).fillna("nan")
        data = np.array(df[sco]).tolist()
        return data


    def block(self, data, blo):
        blodic = {}
        blodiccnt = 0
        for i in range(len(data)):
            if (pd.isna(data[i][blo])):
                if (blodic.__contains__("nan")):
                    self.blocked_list[blodic["nan"]].append(i)
                else:
                    blodic.setdefault("nan", blodiccnt)
                    self.blocked_list.append([])
                    self.blocked_list[blodic["nan"]].append(i)
                    blodiccnt += 1
                    # continue
            elif (blodic.__contains__(data[i][blo])):
                # print(lii[blodic[li[i][blo]]])
                self.blocked_list[blodic[data[i][blo]]].append(i)
            else:
                blodic.setdefault(data[i][blo], blodiccnt)
                self.blocked_list.append([])
                self.blocked_list[blodic[data[i][blo]]].append(i)
                blodiccnt += 1
        # print(blocked_list)
        return self.blocked_list

    def block_con(self, data, j, blo):
        blodic = {}
        blodiccnt = 0
        for i in range(len(data)):
            if (pd.isna(data[i][j])):
                continue
            elif (data[i][j] == blo):
                self.blocked_list[0].append(i)
        self.blocked_list.append([])
        # print(blocked_list)
        return self.blocked_list
   

    def iterate(self, blocked_list):
        pair = [[]]
        for i in range(len(blocked_list)):
            for j in range(len(blocked_list[i])):
                for k in range(j + 1, len(blocked_list[i])):
                    pair[i].append([blocked_list[i][j], blocked_list[i][k]])
            pair.append([])
        # print(pair)
        return pair

    

    def generate(self, index):
        biaodashi = []
        bds = ""
        for predicate in self.Rules[index].predicates:
            if (predicate.property[0] == "constant" or predicate.property[1] == "constant"):
                if (predicate.opt == '!='):
                    biaodashi.append("str(data[l][self.attr_index[\"" + str(predicate.components[0]) + "\"]])" + "!=" + "str(\"" + str(predicate.components[1]) + "\")")
                elif(predicate.opt == '='):
                    biaodashi.append("str(data[l][self.attr_index[\"" + str(
                        predicate.components[0]) + "\"]])" + "==" + "str(\"" + str(predicate.components[1]) + "\")")
                else:
                    biaodashi.append("float(data[l][self.attr_index[\"" + str(predicate.components[0]) + "\"]])" + predicate.opt + "float(" + str(predicate.components[1]) + ")")
            else:
                if (predicate.opt == '!='):
                    biaodashi.append("str(data[l][self.attr_index[\"" + str(predicate.components[0]) + "\"]])" + "!=" + "str(data[r][self.attr_index[\"" + str(predicate.components[1]) + "\"]])")
                elif(predicate.opt == '='):
                    biaodashi.append("str(data[l][self.attr_index[\"" + str(
                        predicate.components[0]) + "\"]])" + "==" + "str(data[r][self.attr_index[\"" + str(
                        predicate.components[1]) + "\"]])")
                else:
                    biaodashi.append("float(data[l][self.attr_index[\"" + str(predicate.components[0]) + "\"]])" + predicate.opt + "float(data[r][self.attr_index[\"" + str(predicate.components[1]) + "\"]])")

        for i in range(len(biaodashi)):
            if (i == 0):
                bds = biaodashi[i]
            else:
                bds = bds + " and " + biaodashi[i]
        return bds


    def ocjoin(self, index, blocked_list, data):
        # default primary attribute is attribute in blocklist
        data_tempuse = data.copy()
        finaltuples = []
        conds = []
        K = [[]]
        Kcnt = 0
        flagg = 0
        for predicate in self.Rules[index].predicates:
            if (predicate.opt != '=' and predicate.opt != '!=' ):
                conds.append([predicate.components[0], predicate.opt])
        for i in range(len(blocked_list)):
            nbparts = int(len(blocked_list[i]) / 3) + 1
            K = [[]]
            k = 0
            Kcnt = 0
            while (k < len(blocked_list[i])):
                try:
                    K[Kcnt] = blocked_list[i][k :k + nbparts]
                except:
                    K[Kcnt] = blocked_list[i][k : len(blocked_list[i])-1]
                k += nbparts
                Kcnt += 1
                K.append([])
            # cond_his  = []     # Setting an history list to avoid computing again
            for kj in range(len(K)):
                cond_his = []
                # self.sorts[cond_idx] = []
                for cond in conds:
                    cond_idx = self.attr_index[cond[0]]
                    if cond_idx in cond_his:
                        continue
                    else:
                        self.sorts[cond_idx] = []
                        cond_his.append(cond_idx)
                    for Kin in range(len(K[kj])):
                        try:
                            data_tempuse[K[kj][Kin]][cond_idx] = float(data_tempuse[K[kj][Kin]][cond_idx])
                        except:
                            data_tempuse[K[kj][Kin]][cond_idx] = 0
                        self.sorts[cond_idx].append([data_tempuse[K[kj][Kin]][cond_idx], Kin])
                    if (cond[1] == '>=' or cond[1] == '>'):
                        self.sorts[cond_idx] = sorted(self.sorts[cond_idx], key=lambda x: x[0], reverse=True)
                    else:
                        self.sorts[cond_idx] = sorted(self.sorts[cond_idx], key=lambda x: x[0], reverse=False)
                for kl in range(kj+1, len(K)):
                    kjmax = float(self.sorts[self.attr_index[conds[0][0]]][len(self.sorts[self.attr_index[conds[0][0]]]) - 1][0])
                    kjmin = float(self.sorts[self.attr_index[conds[0][0]]][0][0])
                    klmax =  - np.inf
                    klmin = np.inf
                    for cond in conds:
                        cond_idx = self.attr_index[cond[0]]
                        for Kin in range(len(K[kl])):
                            try:
                                data_tempuse[K[kl][Kin]][cond_idx] = float(data_tempuse[K[kl][Kin]][cond_idx])
                            except:
                                data_tempuse[K[kl][Kin]][cond_idx] = 0
                    for Kin in range(len(K[kl])):
                        klmax = max(float(klmax) , float(data_tempuse[K[kl][Kin]][self.attr_index[conds[0][0]]]))
                        klmin = min(float(klmin) , float(data_tempuse[K[kl][Kin]][self.attr_index[conds[0][0]]]))
                    if ((kjmax < klmin) or (kjmin > klmax)):
                        continue
                    else:
                        tuples = []
                        for cond in conds:
                            temptuples = []
                            for klindex in K[kl]:
                                for sort_idx in self.sorts[self.attr_index[cond[0]]]:
                                    bds = "float(sort_idx[0]) " + str(cond[1]) + " float(data_tempuse[klindex][self.attr_index[cond[0]]])"
                                    if(eval(bds)):
                                        try:
                                            temptuples.append((K[kj][sort_idx[1]] , klindex))
                                        except:
                                            print("kj:",kj,"sort_idx[1]",sort_idx[1])
                                    else:
                                        continue
                            if (tuples == []):
                                tuples = temptuples
                            else:
                                for pair1 in tuples:
                                    if (pair1 in temptuples):
                                        flagg = 1
                                    else:
                                        tuples.remove(pair1)
                                    if (flagg == 1):
                                        flagg = 0
                                        break
                        finaltuples.extend(tuples)
        anotemdic = {"=": 0, "!=": 1, "<": 2, ">": 3, "<=": 4, ">=": 5}
        for pair in finaltuples:
            biaodashi = self.generate(index)
            l = pair[0]
            r = pair[1]
            if (eval(biaodashi)):
                self.vio.append([index])
                for predicate in self.Rules[index].predicates:
                    if (predicate.property[0] == "constant" or predicate.property[1] == "constant"):
                        self.vio[self.cnt].append(anotemdic[predicate.opt])
                        self.vio[self.cnt].append((l, self.attr_index[predicate.components[0]]))
                        self.vio[self.cnt].append(predicate.components[1])
                    else:
                        self.vio[self.cnt].append(anotemdic[predicate.opt])
                        self.vio[self.cnt].append((l, self.attr_index[predicate.components[0]]))
                        self.vio[self.cnt].append((r, self.attr_index[predicate.components[1]]))
                self.cnt += 1

    
    def detect(self, maypair, data):
        
        print("Detecting Errors")
        for i in tqdm(range(len(maypair)-1),ncols=90):
            anotemdic = {"=": 0, "!=": 1, "<": 2, ">": 3, "<=": 4, ">=": 5}
            biaodashi = self.generate(i)
            for j in range(len(maypair[i])):
                for k in maypair[i][j]:
                    l = k[0]
                    r = k[1]
                    try:
                        eval(biaodashi)
                    except:
                        for predicate in self.Rules[i].predicates:
                            if (predicate.property[0] == "constant" or predicate.property[1] == "constant"):
                                pass
                            else:
                                if (predicate.opt != '=' and predicate.opt != '!='):
                                    try:
                                        float(data[l][self.attr_index[predicate.components[0]]])
                                    except:
                                        data[l][self.attr_index[predicate.components[0]]] = 0
                                    try:
                                        float(data[r][self.attr_index[predicate.components[1]]])
                                    except:
                                        data[r][self.attr_index[predicate.components[1]]] = 0

                    if (eval(biaodashi)):
                        self.vio.append([i])
                        for predicate in self.Rules[i].predicates:
                            if (predicate.property[0] == "constant" or predicate.property[1] == "constant"):
                                self.vio[self.cnt].append(anotemdic[predicate.opt])
                                self.vio[self.cnt].append((l, self.attr_index[predicate.components[0]]))
                                self.vio[self.cnt].append(predicate.components[1])
                            else:
                                self.vio[self.cnt].append(anotemdic[predicate.opt])
                                self.vio[self.cnt].append((l, self.attr_index[predicate.components[0]]))
                                self.vio[self.cnt].append((r, self.attr_index[predicate.components[1]]))
                        self.cnt += 1
        return self.vio

    def repair(self, data, vio):
        sizebefore = 0
        sizeafter = 0
        processedcell = []
        
        input_data = read_graph_dc(vio)
        dicc = input_data.copy()
        for i in dicc:
            dicc[i] = list(set(dicc[i]))
        for i in dicc.items():
            processedcell.append(i[0])
        sizebefore = len(processedcell)
        self.repaired_cells = []
        self.clean_in_cands = []
        self.clean_in_cands_repair_right = []
        self.repair_right_cells = []
        self.repaired_cells_value = {}
        while (sizebefore > sizeafter):
            sizebefore = len(processedcell)
            input_data = read_graph_dc(vio)
      
            dicc = input_data.copy()
            
            for i in dicc:
                dicc[i] = list(set(dicc[i]))
            
            diccop = copy.deepcopy(dicc)
            # 
            self.mvc = greedy_min_vertex_cover_dc(dicc, vio)
            mvcdic = copy.deepcopy(self.mvc)
            while self.mvc:
                cell = self.mvc.pop()
                if PERFECTED and not ONLYED:
                    if (cell[0], self.schema.index(list(self.attr_index.keys())[cell[1]])) not in self.wrong_cells:
                        continue
                # print("cell:",cell)
                edges = dicc[cell]
                while edges:
                    edge = edges.pop()
                    index1 = vio[edge].index(cell)
       
                    if (index1 % 3 == 2):
                        index2 = index1 + 1
                        index0 = index1 - 1
                    if (index1 % 3 == 0):
                        index2 = index1 - 1
                        index0 = index1 - 2
                    if (index1 % 3 == 1):
                        continue
                    self.visdic.clear()
                    self.edgedic.clear()
                    # 
                    exps = []
                    exps = self.lookup(cell, vio[edge][index2], vio[edge][index0], diccop, mvcdic, vio, cell)
                l = cell[0]
                rr = cell[1]
                repair_cands = self.determination(exps, data)
                try:
                    truerepair = repair_cands[0]
                    if str(self.data_cl[l][rr]) in repair_cands:
                        self.clean_in_cands.append((l, self.schema.index(list(self.attr_index.keys())[rr])))
                        if ((str(truerepair) == str(self.data_cl[l][rr]))):
                            self.clean_in_cands_repair_right.append((l, self.schema.index(list(self.attr_index.keys())[rr])))
                except:
                    truerepair = 0
                self.exps.clear()
                # 
                self.all_clean = self.all_clean + 1
                self.repaired_cells.append((l, self.schema.index(list(self.attr_index.keys())[rr])))
                self.repaired_cells_value[(l, self.schema.index(list(self.attr_index.keys())[rr]))] = truerepair
                # 
                if ((str(truerepair) == str(self.data_cl[l][rr]))):
                    self.clean_right_pre = self.clean_right_pre + 1
                    self.repair_right_cells.append((l, self.schema.index(list(self.attr_index.keys())[rr])))
                if ((str(truerepair) == str(self.data_cl[l][rr])) and (data[l][rr] != self.data_cl[l][rr])):
                    self.clean_right = self.clean_right + 1
                data[l][rr] = truerepair
            print("Finish Repairing")
            vio = self.detect(self.maypair, data)
            input_data = read_graph_dc(vio)
            # 
            dicc = input_data.copy()
            if (len(list(dicc)) == 0):
                return data, self.all_clean, self.clean_right
            for i in dicc.items():
                processedcell.append(i[0])
            sizeafter = len(processedcell)
        self.all_clean, self.clean_right, self.clean_right_pre = self.postprocess(self.mvc, dicc, self.all_clean, self.clean_right, self.clean_right_pre)
        return data, self.all_clean, self.clean_right, self.clean_right_pre


    def postprocess(self, mvc, dicc, all_clean, clean_right, clean_right_pre):
        while mvc:
            cell = mvc.pop()
            edges = dicc[cell]
            edge = edges.pop()
            index1 = self.vio[edge].index(cell)
            if (index1 % 3 == 2):
                index2 = index1 + 1
                index0 = index1 - 1
            if (index1 % 3 == 0):
                index2 = index1 - 1
                index0 = index1 - 2
            if (index1 % 3 == 1):
                continue
            l1 = cell[0]
            r1 = cell[1]
            l2 = self.vio[edge][index2][0]
            r2 = self.vio[edge][index2][1]
            if (self.vio[edge][index0] == 1):
                truerepair = self.data[l2][r2]
            if (self.vio[edge][index0] == 2):
                truerepair = self.data[l2][r2] - 1
            if (self.vio[edge][index0] == 3):
                truerepair = self.data[l2][r2] + 1
            if (self.vio[edge][index0] == 4):
                truerepair = self.data[l2][r2]
            if (self.vio[edge][index0] == 5):
                truerepair = self.data[l2][r2]
            all_clean += 1
            self.repaired_cells.append((l1, self.schema.index(list(self.attr_index.keys())[r1])))
            self.repaired_cells_value[(l1, self.schema.index(list(self.attr_index.keys())[r1]))] = truerepair
            if (str(truerepair) == str(self.data_cl[l1][r1])):
                clean_right_pre = clean_right_pre + 1
                self.repair_right_cells.append((l1, self.schema.index(list(self.attr_index.keys())[r1])))
            if ((str(truerepair) == str(self.data_cl[l1][r1])) and (self.data[l1][r1] != self.data_cl[l1][r1])):
                clean_right = clean_right + 1
            self.data[l1][r1] = truerepair
        return all_clean, clean_right, clean_right_pre

    def lookup(self, cell, edge, oper, diccop, mvcdic, vio, firstcell):
        # 
        if (firstcell != cell):
            return []
        self.exps.extend([[cell, edge, oper]])
        # 
        front = []
        # print(diccop[cell])
        try:
            front.extend(diccop[cell])
        except:
            pass
        for i in front:
            index1 = vio[i].index(cell)
            if (index1 % 3 == 2):
                index2 = index1 + 1
                index0 = index1 - 1
            if (index1 % 3 == 0):
                index2 = index1 - 1
                index0 = index1 - 2
            if (mvcdic.__contains__(vio[i][index2]) and vio[i][index2] != cell):
                continue
            if (self.visdic.__contains__(vio[i][index2])):
                continue
            # if (edgedic.__contains__(i)):
            #     continue
            # edgedic[i] = 1
            self.visdic[vio[i][index2]] = 1
            try:
                edges = diccop[vio[i][index2]]
            except:
                edges = []
                continue
            # print("edges:",edges)
            for j in edges:
                index11 = vio[j].index(vio[i][index2])
                if (index11 % 3 == 2):
                    index22 = index11 + 1
                    index00 = index11 - 1
                if (index11 % 3 == 0):
                    index22 = index11 - 1
                    index00 = index11 - 2
                # print("1:",vio[i][index2],"index11:",index11,"index22:",index22,"2:",vio[i][index22])
                if (mvcdic.__contains__(vio[j][index22])):
                    continue
                if (self.visdic.__contains__(vio[j][index22])):
                    continue
                # if (edgedic.__contains__(j)):
                #     continue
                self.visdic[vio[j][index22]] = 1
                # edgedic[j] = 1
                self.exps.extend(self.lookup(vio[i][index2], vio[j][index22], vio[j][index00], diccop, mvcdic, vio, cell))
        return self.exps

    def determination(self, exps, data):
        opt_dict = ["==", "!=", "<", ">", "<=", ">="]
        rep_dict = {}
        rep_dict.clear()
        max = -np.inf
        min = np.inf
        finalthing = -2
        for i in exps:
            temp1 = i[1][0]
            temp2 = i[1][1]
            # print(i[2])
            if (opt_dict[i[2]] == "=="):
                continue
            if (opt_dict[i[2]] == "!="):
                if (rep_dict.__contains__(data[temp1][temp2])):
                    # print(1)
                    rep_dict[data[temp1][temp2]] = rep_dict[data[temp1][temp2]] + 1
                else:
                    rep_dict[data[temp1][temp2]] = 1
            else:
                # f=np.append(f, -2*data[temp1][temp2])
                if (opt_dict[i[2]] == "<" or opt_dict[i[2]] == "<="):
                    if (max < float(data[temp1][temp2])):
                        max = float(data[temp1][temp2])
                    finalthing = max + 1
                else:
                    if (min > float(data[temp1][temp2]) and data[temp1][temp2] != 0):
                        min = float(data[temp1][temp2])
                    finalthing = min - 1

        if (finalthing != -2):
            return finalthing
        sorted(rep_dict.items(), key=lambda x: x[1])
        rep_final = list(rep_dict)
        try:
            return rep_final
        except:
            return 0
    # @profile
    def run(self, file_path, dirty_path, clean_path):
        tracemalloc.start()
        # os.chdir(path)
        #
        df = pd.read_csv(dirty_path).astype(str).fillna("nan")
        self.schema = list(df.columns)
        f = open(clean_path, "r")
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            file = f.read()
        rules = file.split('\n')
        # 
        sco = []
        diccnt = 0
        flag = 0
        # 
        equal_num = 0
        other_num = 0

        #
        for rule in rules:
            equal_num = 0
            other_num = 0
            DR = DCRule(rule, self.schema)
            for predicate in DR.predicates:
                if (predicate.property[0] == "attribute"):
                    sco.append(predicate.components[0])
                if (predicate.property[1] == "attribute"):
                    sco.append(predicate.components[1])
                if (predicate.opt == '='):
                    equal_num += 1
                else:
                    other_num += 1
            DR_copy = copy.deepcopy(DR)
            DR_copy.att_copy(DR)
            self.Rules.append(DR_copy)
        attr_indexcnt = 0
        for i in range(len(sco)):
            sco[i] = sco[i].strip()
        sco = list(set(sco))
        for i in sco:
            self.sorts.append([])
            self.attr_index.setdefault(i, attr_indexcnt)
            attr_indexcnt += 1

        data = self.scope(sco)
        # all_wrong的计算
        all_wrong = 0
        self.data_cl = self.scope1(sco)
        # print(li_cl)
        dirty_df = pd.read_csv(dirty_path, header=0).astype(str).fillna("nan")
        clean_df = pd.read_csv(clean_path, header=0).astype(str).fillna("nan")
        data_wrong = np.array(dirty_df).tolist()
        df = pd.read_csv(clean_path, header=0).astype(str).fillna("nan")
        data_clean = np.array(df).tolist()
        # 
        self.wrong_cells = []
        for i in range(len(dirty_df)):
            for j in range(len(dirty_df.columns)):
                if dirty_df.iloc[i, j] != clean_df.iloc[i, j]:
                    self.wrong_cells.append((i, j))
        # for i in range(len(data_wrong)):
        #     for j in range(len(data_wrong[i])):
        #         try:
        #             float(data_wrong[i][j])
        #             float(data_clean[i][j])
        #             if (float(data_wrong[i][j]) != float(data_clean[i][j])):
        #                 all_wrong += 1
        #                 self.wrong_cells.append((i,j))
        #                 # print("Wrong:", data_wrong[i][j], "clean:", data_clean[i][j])
        #         except:
        #             if (str(data_wrong[i][j]) != str(data_clean[i][j])):
        #                 all_wrong += 1
        #                 self.wrong_cells.append((i,j))
                        # print("Wrong:", data_wrong[i][j], "clean:", data_clean[i][j])
        # print("all_wrong:", all_wrong)
        # 
        print("Start Blocking")
        icnt = 0
        for i in tqdm(range(len(self.Rules)),ncols=90):
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            i_equalnum = 0
            i_othernum = 0
            i_ocjoinnum = 0
            equal_components = []
            constantflag = 0
            constant_pre = []
            con_equal_components = []
            for predicate in self.Rules[i].predicates:
                flag = 0
                if (predicate.property[0] == "constant" or predicate.property[1] == "constant"):
                    constantflag = 1
                    self.constant_pre.append(predicate)
                    if (predicate.opt == '='):
                        con_equal_components.append(predicate.components[0])
                        self.contantdic.setdefault(predicate.components[0], predicate.components[1])
                if (predicate.opt == '=' and predicate.property[0] == "attribute" and predicate.property[
                    1] == "attribute"):
                    equal_components.append(predicate.components[0])
                    i_equalnum += 1
                else:
                    i_othernum += 1
                    if (predicate.opt != '!=' and predicate.property[0] == "attribute" and predicate.property[
                    1] == "attribute"):
                        i_ocjoinnum += 1
            if (constantflag == 1):
                if (i_othernum == 0 and i_equalnum == 0):
                    flag = self.con_equal_compts(sco, flag, data, icnt, con_equal_components)
                elif (i_othernum == 0 and i_equalnum != 0):
                    flag = self.con_equal_compts(sco, flag, data, icnt, con_equal_components)
                    flag = self.equal_compts(sco, flag, data, icnt, equal_components)
                elif (i_othernum != 0 and i_equalnum == 0):
                    flag = self.con_equal_compts(sco, flag, data, icnt, con_equal_components)
                elif (i_othernum != 0 and i_equalnum != 0):
                    if (i_ocjoinnum == 0):
                        flag = self.con_equal_compts(sco, flag, data, icnt, con_equal_components)
                        flag = self.equal_compts(sco, flag, data, icnt, equal_components)
                    else:
                        for j in range(len(sco)):
                            for k in range(len(equal_components)):
                                if (equal_components[k] == sco[j]):
                                    self.blocked_list = []
                                    self.block(data, j)
                                    self.ocjoin(i, self.blocked_list, data)
                    for i in range(len(self.sorts)):
                        self.sorts[i].clear()
                icnt += 1
                self.maypair.append([])
            elif (len(self.Rules[i].variable) == 1):
                if i_equalnum == 0:
                    for j in range(len(data)):
                        self.maypair[icnt].append([j, j])
                else:
                    flag = self.equal_compts(sco, flag, data, icnt, equal_components)
                icnt += 1
                self.maypair.append([])
            elif (i_othernum == 0): # whatever i_equalnum is 0 or not
                flag = self.equal_compts(sco, flag, data, icnt, equal_components)
                icnt += 1
                self.maypair.append([])
            elif (i_equalnum == 0 and i_othernum != 0):
                if (i_ocjoinnum == 0):
                    self.blocked_list = [[]]
                    for j in range(len(data)):
                        self.blocked_list[0].append(j)
                    self.maypair[icnt].extend(self.iterate(self.blocked_list))
                    icnt += 1
                    self.maypair.append([])
                else:
                    self.blocked_list = [[]]
                    for j in range(len(data)):
                        self.blocked_list[0].append(j)
                    self.ocjoin(i, self.blocked_list, data)
            elif (i_equalnum != 0 and i_othernum != 0):
                if (i_ocjoinnum == 0):
                    flag = self.equal_compts(sco, flag, data, icnt, equal_components)
                    icnt += 1
                    self.maypair.append([])
                else:
                    for j in range(len(sco)):
                        for k in range(len(equal_components)):
                            if (equal_components[k] == sco[j]):
                                self.blocked_list = []
                                self.block(data, j)
                                self.ocjoin(i, self.blocked_list, data)
                    # icnt += 1
                # for i in range(len(self.sorts)):
                #     self.sorts[i].clear()

        viocntnt = 0
        print("Finish Blocking and Iterating")
        self.vio = self.detect(self.maypair, data)
        print("Finish Detectings")
        liclean, all_clean, clean_right, clean_right_pre = self.repair(data, self.vio)
        print("Finish Repairing")
        # print("all_clean:", all_clean)
        # print("clean_right:", clean_right)
        # print("clean_right_pre:", clean_right_pre)

        if True:
            self.repaired_cells = list(set(self.repaired_cells))
            self.wrong_cells = list(set(self.wrong_cells))
            if not PERFECTED:
                det_right = 0
                file_name = dirty_path.split('/')[-1].split('.')[0]
                out_path = "./Exp_result/bigdansing/" + task_name[:-1] +"/onlyED_" + task_name + file_name + ".txt"
                if not os.path.exists(os.path.dirname(out_path)):
                    os.makedirs(os.path.dirname(out_path))
                f = open(out_path, 'w')
                # sys.stdout = f
                end_time = time.time()
                for cell in self.repaired_cells:
                    if cell in self.wrong_cells:
                        det_right = det_right + 1
                pre = det_right / (len(self.repaired_cells)+1e-10)
                rec = det_right / (len(self.wrong_cells)+1e-10)
                f1 = 2*pre*rec/(pre+rec+1e-10)
                print("{pre}\n{rec}\n{f1}\n{time}".format(pre=pre, rec=rec, f1=f1, time=(end_time-start_time)))
                f.close()

                rep_right = 0
                rep_total = len(self.repaired_cells)
                wrong_cells = len(self.wrong_cells)
                rec_right = 0
                for cell in self.repair_right_cells:
                    rep_right += 1
                for cell in self.wrong_cells:
                    if cell in self.repair_right_cells:
                        rec_right += 1
                pre = rep_right / (rep_total+1e-10)
                rec = rec_right / (wrong_cells+1e-10)
                f1 = 2*pre*rec / (rec+pre+1e-10)
                out_path = "./Exp_result/bigdansing/" + task_name[:-1] +"/oriED+EC_" + task_name + file_name + ".txt"
                res_path = "./Repaired_res/bigdansing/" + task_name[:-1] +"/repaired_" + task_name + file_name + ".csv"
                if not os.path.exists(os.path.dirname(res_path)):
                    os.makedirs(os.path.dirname(res_path))
                dirty_df = pd.read_csv(dirty_path).astype(str).fillna("nan")
                for cell, value in self.repaired_cells_value.items():
                    dirty_df.iloc[cell[0], cell[1]] = value
                dirty_df.to_csv(res_path, index=False)
                repair_df = copy.deepcopy(dirty_df)
                f = open(out_path, 'w')
                # sys.stdout = f
                print("{pre}\n{rec}\n{f1}\n{time}".format(pre=pre, rec=rec, f1=f1, time=(end_time-start_time)))
                f.close()

                # sys.stdout = sys.__stdout__
                out_path = "./Exp_result/bigdansing/" + task_name[:-1] +"/all_computed_" + task_name + file_name + ".txt"
                f = open(out_path, 'w')
                # sys.stdout = f
                right2wrong = 0
                right2right = 0
                wrong2right = 0
                wrong2wrong = 0

                rep_total = len(self.repaired_cells)
                wrong_cells = len(self.wrong_cells)
                for cell in self.repair_right_cells:
                    if cell in self.wrong_cells:
                        wrong2right += 1
                    else:
                        right2right += 1
                print("rep_right:"+str(rep_right))
                print("rec_right:"+str(rec_right))
                print("wrong_cells:"+str(wrong_cells))
                print("prec:"+str(pre))
                print("rec:"+str(rec))
                print("wrong2right:"+str(wrong2right))
                print("right2right:"+str(right2right))
                self.repair_wrong_cells = [i for i in self.repaired_cells if i not in self.repair_right_cells]
                for cell in self.repair_wrong_cells:
                    if cell in self.wrong_cells:
                        wrong2wrong += 1
                    else:
                        right2wrong += 1
                print("wrong2wrong:"+str(wrong2wrong))
                print("right2wrong:"+str(right2wrong))
                print("proportion of clean value in candidates:"+str(len(self.clean_in_cands)/rep_total))
                print("proportion of clean value in candidates and selected correctly:"+str(len(self.clean_in_cands_repair_right)/len(self.clean_in_cands)))
                f.close()
                return repair_df
                
            else:
                end_time = time.time()
                rep_right = 0
                rep_total = len(self.repaired_cells)
                wrong_cells = len(self.wrong_cells)
                rec_right = 0
                rep_t = 0
                for cell in self.wrong_cells:
                    if cell in self.repaired_cells:
                        rep_t += 1
                        if cell in self.repair_right_cells:
                            rec_right += 1
                pre = rec_right / (rep_t+1e-10)
                rec = rec_right / (wrong_cells+1e-10)
                f1 = 2*pre*rec / (rec+pre+1e-10)
                out_path = "./Exp_result/bigdansing/" + task_name[:-1] +"/perfectED+EC_" + task_name + file_name + ".txt"
                res_path = "./Repaired_res/bigdansing/" + task_name[:-1] +"/perfect_repaired_" + task_name + file_name + ".csv"
                
                
                dirty_df = pd.read_csv(dirty_path).astype(str).fillna("nan")
                for cell, value in self.repaired_cells_value.items():
                    dirty_df.iloc[cell[0], cell[1]] = value
                dirty_df.to_csv(res_path, index=False)
                repair_df = copy.deepcopy(dirty_df)
                f = open(out_path, 'w')
                # sys.stdout = f
                print("{pre}\n{rec}\n{f1}\n{time}".format(pre=pre, rec=rec, f1=f1, time=(end_time-start_time)))
                f.close()
                return repair_df

    def equal_compts(self, sco, flag, data, icnt, equal_components):
        for j in range(len(sco)):
            for k in range(len(equal_components)):
                if (equal_components[k] == sco[j]):
                    self.blocked_list = []
                    self.block(data, j)
                    if ((len(equal_components) > 1) and flag == 1):
                        newli = self.iterate(self.blocked_list)
                        for pairlist in self.maypair[icnt]:
                            flagg = 0
                            for pair1 in pairlist:
                                for pairlist2 in newli:
                                    if (pair1 in pairlist2):
                                        flagg = 1
                                    if (flagg == 1):
                                        flagg = 0
                                        break
                                    if (pairlist2 == newli[-1]):
                                        pairlist.remove(pair1)
                                        break
                            pass
                    else:
                        flag = 1
                        self.maypair[icnt].extend(self.iterate(self.blocked_list))
        return flag

    def con_equal_compts(self, sco, flag, data, icnt, con_equal_components):
        for j in range(len(sco)):
            for k in range(len(con_equal_components)):
                if (con_equal_components[k] == sco[j]):
                    self.blocked_list = [[]]
                    self.block_con(data, j, self.contantdic[sco[j]])
                    if ((len(con_equal_components) > 1) and flag == 1):
                        newli = self.iterate(self.blocked_list)
                        for pairlist in self.maypair[icnt]:
                            for pair1 in pairlist:
                                for pairlist2 in newli:
                                    if (pair1 in pairlist2):
                                        flagg = 1
                                    if (flagg == 1):
                                        flagg = 0
                                        break
                                    if (pairlist2 == newli[-1]):
                                        pairlist.remove(pair1)
                                        break
                            pass
                    else:
                        flag = 1
                        self.maypair[icnt].extend(self.iterate(self.blocked_list))
        return flag

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#####################################################################
import yaml
config_path = sys.argv[1]
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# config = yaml.safe_load(sys.argv[3]) 
rule_path = config['rule_path']
dirty_path = config['dirty_path']
clean_path = config['clean_path']
task_name = config['task_name']
ONLYED = 0
PERFECTED = 0


time_limit = 24*3600
signal.signal(signal.SIGALRM, handler)
signal.alarm(time_limit)
try:
    start_time = time.time()
    bd = BigDansing()
    repair_df = bd.run(rule_path, dirty_path, clean_path)
except TimeoutError as e: 
    print("Time exceeded:", e, task_name, dirty_path)
    out_file = open("./aggre_results/timeout_log.txt", "a")
    now = datetime.now()
    out_file.write(now.strftime("%Y-%m-%d %H:%M:%S"))
    out_file.write(" BigDansing.py: ")
    out_file.write(f" {task_name}")
    out_file.write(f" {dirty_path}\n")
    out_file.close()

# bigdancing_res_path = 'Repaired_res/bigdancing_hos_0306.csv'
bigdancing_res_path = config['bigdancing_res_path']
# bigdancing_res_path = sys.argv[2]
repair_df.to_csv(bigdancing_res_path, index=False)
print('bigdancing is ok!')
