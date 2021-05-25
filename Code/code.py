import time

import MySQLdb
import math
import numpy as np


class CasualExplore(object):
    def __init__(self, target, limit, topK, alpha_Relation, alpha_Wc, beta_Wc, omega_Wc, lamda, Fai, fai, d_PageRank,
                 score_Alpha, score_Beta, score_Omega):
        self.cursor = MySQLdb.connect(user='root', passwd='1234', charset='utf8', db='dbpedia2016').cursor()
        self.target = target
        self.limit = limit
        self.alpha_Relation = alpha_Relation
        self.alpha_Wc, self.beta_Wc, self.omega_Wc = alpha_Wc, beta_Wc, omega_Wc
        self.lamda = lamda
        self.Fai = Fai
        self.fai = fai
        self.d_PageRank = d_PageRank
        self.TopK = topK
        self.score_Alpha, self.score_Beta, self.score_Omega = score_Alpha, score_Beta, score_Omega
        self.entitySet = set()
        self.targetClassList = self.getClassByEntity(target)
        self.searchedDictByEntity = {}
        self.searchedEntityList = [target]
        self.searchedDictByPathLength = {0: [target]}
        self.classSet = set(self.getClassByEntity(target))
        self.classEntityDict = {class_: [target] for class_ in self.getClassByEntity(target)}
        self.entityNeighbor = {}
        self.classRelationList = []
        self.classIncidentList = []
        self.w1List = []
        self.w2List = []
        self.w3List = []
        self.wcList = []
        self.EntityRelationDict = {}
        self.getC2Height = []
        self.getC2Depth = []
        self.ClassM = np.matrix
        self.EntityM = np.matrix
        self.ClassPR = []
        self.ClassPRList = []
        self.EntityPR = []
        self.EntityPRList = []
        self.TF_IDF = []
        self.EntityPopList = []

    def getClassByEntity(self, target):
        if "'" in target or '"' in target:
            target = target.replace("'", "''").replace('"', '""')
        sql = "select type from type_transitive_yago_new where entity ='{}' group by type".format(target)
        self.cursor.execute(sql)
        this_answer = self.cursor.fetchall()
        class_List = []
        for class_ in this_answer:
            if "Wikicat" not in class_[0]:
                class_List.append(class_[0])
        return class_List

    def entityIn(self, target):
        if "'" in target or '"' in target:
            target = target.replace("'", "''").replace('"', '""')
        sql = "select s from triples_dbp_new where o='{}'".format(target)
        self.cursor.execute(sql)
        entityInList = [_[0] for _ in self.cursor.fetchall()]
        return entityInList

    def entityOut(self, target):
        if "'" in target or '"' in target:
            target = target.replace("'", "''").replace('"', '""')
        sql = "select o from triples_dbp_new where s='{}'".format(target)
        self.cursor.execute(sql)
        entityOutList = [_[0] for _ in self.cursor.fetchall()]
        return entityOutList

    def mergeEntityList(self, target):
        entityInList, entityOutList = self.entityIn(target), self.entityOut(target)
        entityList = entityOutList.copy()
        for _ in entityInList:
            if _ not in entityList:
                entityList.append(_)
        return entityList

    def bfs(self):
        self.entitySet.add(self.target)
        current = 1
        while current <= self.limit:
            nextList = []
            print("现在要搜索的实体总数为", len(self.searchedEntityList))
            for entity in self.searchedEntityList:
                nextList.extend(self._bfs(entity))
            self.searchedDictByPathLength[current] = nextList
            current += 1
            self.searchedEntityList = nextList
        self.classSet = sorted(self.classSet)
        self.entitySet = sorted(self.entitySet)

    def _bfs(self, this_Entity):
        entityList = self.mergeEntityList(this_Entity)
        if "City108524735" in self.getClassByEntity(this_Entity):
            return []
        for entity in entityList:
            classList = self.getClassByEntity(entity)
            if len(classList) != 0:
                self.entitySet.add(entity)
                for class_ in classList:
                    if class_ not in self.targetClassList:
                        self.classSet.add(class_)
                    else:
                        class_ = "Similar" + class_
                        self.classSet.add(class_)
                    if class_ not in self.classEntityDict:
                        self.classEntityDict[class_] = [entity]
                    else:
                        if entity not in self.classEntityDict[class_]:
                            self.classEntityDict[class_].append(entity)
        return entityList

    def getEntityConnectedLength(self, vertexEntityFrom, vertexEntityTo):

        searched_List = [vertexEntityFrom]
        index = 1
        while index <= self.limit:
            next_List = []
            for entity in searched_List:
                if entity == vertexEntityTo:
                    return index
                else:
                    entity_list = self.entityOut(entity)
                    next_List.extend(entity_list)
            searched_List = next_List
            index += 1
        return float('inf')

    def getRelationOfClass(self):
        self.bfs()
        self.getAllEntityNeighbor()
        for iex, class_ in enumerate(self.classSet):
            print("已经计算到第{}".format(iex + 1), " 共{}条".format(len(self.classSet)))
            start = time.time()
            RelationList = []
            for class__ in self.classSet:
                if class_ != class__:
                    RelationList.append(self.getLengthListOfClass(class_, class__))
                else:
                    RelationList.append(0.0)
            self.classRelationList.append(RelationList)
            end = time.time()
            print("该类找到所有relation耗用了{}".format((end - start)))

    def getLengthListOfClass(self, vertexClassFrom, vertexClassTo):
        vertexEntityFromList = self.classEntityDict[vertexClassFrom]
        vertexEntityToList = self.classEntityDict[vertexClassTo]
        lengthList = []
        for vertexEntityFrom in vertexEntityFromList:
            vertexEntityFromDict = self.entityNeighbor[vertexEntityFrom]
            lengthOneList = vertexEntityFromDict[1]
            lengthTwoList = vertexEntityFromDict[2]
            for vertexEntityTo in vertexEntityToList:
                if vertexEntityTo in lengthOneList:
                    lengthList.append(1)
                else:
                    if vertexEntityTo in lengthTwoList:
                        lengthList.append(2)
                    else:
                        lengthList.append(float('inf'))
        Relation = sum([pow(self.alpha_Relation, i) for i in lengthList]) / len(lengthList)
        return Relation

    def getAllEntityNeighbor(self):
        for entity in self.entitySet:
            search_List = [entity]
            index = 0
            entityNeighborDict = {1: [], 2: []}
            while index < self.limit:
                next_List = []
                for searchEntity in search_List:
                    entity_List = self.entityOut(searchEntity)
                    entityNeighborDict[index + 1].extend(entity_List)
                    self.entityNeighbor[entity] = entityNeighborDict
                    next_List.extend(entity_List)
                search_List = next_List
                index += 1

    def getDiscrepancy(self, c1, c2):
        c1_List = self.classEntityDict[c1]
        c2_List = self.classEntityDict[c2]
        c1_set = set(c1_List)
        c2_set = set(c2_List)
        return 1 - len(c1_set & c2_set) / len(c1_set | c2_set)

    def getHierarchyHeight_Top(self, this_Class):
        self.getC2Height = []
        self._getHierarchyHeight_Top(this_Class, 0)
        return max(self.getC2Height)

    def _getHierarchyHeight_Top(self, this_Class, Height):
        if "'" in this_Class or '"' in this_Class:
            this_Class = this_Class.replace("'", "''").replace('"', '""')
        if "Similar" in this_Class:
            this_Class = this_Class.replace("Similar", "")
        sql = "select superclass from class_hierarchy_yago_new where class='{}'".format(this_Class)
        self.cursor.execute(sql)
        rowCount = self.cursor.rowcount
        if rowCount == 0:
            self.getC2Height.append(Height)
            return
        else:
            answer = self.cursor.fetchall()
            for _answer in answer:
                self._getHierarchyHeight_Top(_answer[0], Height + 1)

    def getHierarchyDepth_Bottom(self, this_Class):
        self.getC2Depth = []
        self._getHierarchyDepth_Bottom(this_Class, 0)
        if len(self.getC2Depth) == 0:
            return 0
        else:
            return max(self.getC2Depth)

    def _getHierarchyDepth_Bottom(self, this_Class, Depth):
        if "'" in this_Class or '"' in this_Class:
            this_Class = this_Class.replace("'", "''").replace('"', '""')
        if "Similar" in this_Class:
            this_Class = this_Class.replace("Similar", "")
        sql = "select class from class_hierarchy_yago_new where superclass='{}'".format(this_Class)
        self.cursor.execute(sql)
        rowCount = self.cursor.rowcount
        if rowCount == 0:
            self.getC2Depth.append(Depth)
            return
        else:
            answer = self.cursor.fetchall()
            for _answer in answer:
                self._getHierarchyDepth_Bottom(_answer[0], Depth + 1)

    def getW1(self, c2):
        return len(self.classEntityDict[c2]) / len(self.entitySet)

    def getW2(self, c2):
        ans = 0
        for s in c2:
            if s.isupper():
                ans += 1
        return math.exp(-abs(ans))

    def getW3(self, c2):
        distance_Top = self.getHierarchyHeight_Top(c2)
        distance_Bottom = self.getHierarchyDepth_Bottom(c2)
        if distance_Top + distance_Bottom == 0:
            return 0
        else:
            return distance_Top / (distance_Top + distance_Bottom)

    def getWc(self, c2):
        indexC2 = list(self.classSet).index(c2)
        return self.alpha_Wc * self.w1List[indexC2] + self.beta_Wc * self.w2List[indexC2] + self.omega_Wc * self.w3List[
            indexC2]

    def getIncidenceConnected(self, relation, discrepancy, c2):
        indexC2 = list(self.classSet).index(c2)
        return self.wcList[indexC2] * (self.lamda * relation + (1 - self.lamda) * discrepancy)

    def getRelateness(self, c1, c2):
        indexC1 = list(self.classSet).index(c1)
        indexC2 = list(self.classSet).index(c2)
        return self.classRelationList[indexC1][indexC2]

    def getIncidenceAll(self, c1, c2):
        relation = self.getRelateness(c1, c2)
        discrepancy = self.getDiscrepancy(c1, c2)
        indexC2 = list(self.classSet).index(c2)
        if relation == 0:
            return self.wcList[indexC2] * discrepancy
        else:
            return self.getIncidenceConnected(relation, discrepancy, c2)

    def getClassIncidence(self):
        for iex, class_ in enumerate(self.classSet):
            print("进行到第{}个".format(iex + 1))
            self.w1List.append(self.getW1(class_))
            self.w2List.append(self.getW2(class_))
            self.w3List.append(self.getW3(class_))
        self.w1List = [k / sum(self.w1List) for k in self.w1List]
        self.w2List = [k / sum(self.w2List) for k in self.w2List]
        self.w3List = [k / sum(self.w3List) for k in self.w3List]
        for class_ in self.classSet:
            self.wcList.append(self.getWc(class_))
        for iex, class_ in enumerate(self.classSet):
            incidence_list = []
            print("正在计算概率--------进行到{}个类".format(iex + 1))
            for class__ in self.classSet:
                incidence_list.append(self.getIncidenceAll(class_, class__))
            self.classIncidentList.append(incidence_list)

    def getClassMatrix(self):
        self.getClassIncidence()
        for i in range(0, len(self.classIncidentList)):
            print("正在构建第{}列类矩阵".format(i + 1))
            self.classIncidentList[i] = [k / sum(self.classIncidentList[i]) for k in self.classIncidentList[i]]
        self.ClassM = np.array(self.classIncidentList)
        self.ClassM = self.ClassM.T
        print("Class矩阵已经构建完成")

    def matrix_Multiply(self, M, p):
        return np.matmul(M, p)

    def getClassPageRank(self):
        self.getClassMatrix()
        n = len(self.classSet)
        self.ClassM = np.mat(self.ClassM)
        M_inverse = (np.identity(n) - self.d_PageRank * self.ClassM).I
        e_T = np.ones(n)
        second = (1 - self.d_PageRank) * e_T / n
        p = self.matrix_Multiply(M_inverse, second)
        self.ClassPR = p

    def displayClassPR(self):
        self.getClassPageRank()
        for index, class_ in enumerate(self.classSet):
            self.ClassPRList.append((class_, list(self.ClassPR[:, [index]].A)[0][0]))
        self.ClassPRList = sorted(self.ClassPRList, key=lambda classPR: classPR[1], reverse=True)
        print("进行类排序后得到的结果为")
        for index, _ in enumerate(self.ClassPRList):
            print("{}. {} PR值为{}".format(index + 1, _[0], _[1]))
        print("下面是根据类排名取出的实体排行")
        self.get_TopK_ByClassPR()
        print("--------------------------------------------------------------------------------------")

    def get_TopK_ByClassPR(self):
        answerList = []
        for i in range(len(self.classSet)):
            entityList = self.classEntityDict[self.ClassPRList[i][0]]
            goodList = []
            for entity in entityList:
                goodList.append((entity, self.getGood(entity)))
            goodList = sorted(goodList, key=lambda goodlist: goodlist[1], reverse=True)
            for item in goodList:
                if item[0] in answerList or item[0] == self.target:
                    continue
                else:
                    answerList.append(item[0])
                    print("{}. {}".format(len(answerList), item[0]))
                    break
        for i in range(0, len(self.TopK)):
            print("topK={}, diversity score={}, coverage score={}, good score={}".format(self.TopK[i],
                                                                                         self.getDiversity(answerList[
                                                                                                           0:self.TopK[
                                                                                                               i]]),
                                                                                         self.getCoverage(answerList[
                                                                                                          0:self.TopK[
                                                                                                              i]]),
                                                                                         self.getGoodness(answerList[
                                                                                                          0:self.TopK[
                                                                                                              i]])))

    def getPop(self, entity):
        neighbor_List = self.entityIn(entity) + self.entityOut(entity)
        neighbor = []
        for i in range(0, len(neighbor_List)):
            if neighbor_List[i] in self.entitySet:
                neighbor.append(neighbor_List[i])
        return len(neighbor) / len(self.entitySet)

    def getGood(self, entity):
        return self.Fai * len(self.getClassByEntity(entity)) / len(self.classSet) + self.fai * self.getPop(entity)

    def getEntityMatrix(self):
        array_List = []
        for node_ in self.entitySet:
            RelationList = []
            for node__ in self.entitySet:
                if node_ == node__:
                    RelationList.append(0.0)
                else:
                    lengthOneList = self.entityNeighbor[node_][1]
                    lengthTwoList = self.entityNeighbor[node_][2]
                    if node__ in lengthOneList:
                        relation = 1.0
                    else:
                        if node__ in lengthTwoList:
                            relation = 2.0
                        else:
                            relation = float('inf')
                    if relation <= self.limit:
                        RelationList.append(1.0)
                    else:
                        RelationList.append(0.0)
            self.EntityRelationDict[node_] = RelationList
        for node in self.entitySet:
            array_List.append(self.EntityRelationDict[node])
        for i in range(0, len(array_List)):
            if sum(array_List[i]) != 0:
                array_List[i] = [k / sum(array_List[i]) for k in array_List[i]]
        self.EntityM = np.array(array_List[0])
        self.EntityM.resize((len(array_List[0]), 1))
        for i in range(1, len(array_List)):
            tmp_Array = np.array(array_List[i])
            tmp_Array.resize((len(array_List[i]), 1))
            self.EntityM = np.hstack((self.EntityM, tmp_Array))

    def getEntityPageRank(self):
        self.getEntityMatrix()
        n = len(self.entitySet)
        self.EntityM = np.mat(self.EntityM)
        M_inverse = (np.identity(n) - self.d_PageRank * self.EntityM).I
        e_T = np.ones(n)
        second = (1 - self.d_PageRank) * e_T / n
        p = self.matrix_Multiply(M_inverse, second)
        self.EntityPR = p

    def displayEntityPR(self):
        self.getEntityPageRank()
        answerList = []
        for index, node_ in enumerate(self.entitySet):
            self.EntityPRList.append((node_, list(self.EntityPR[:, [index]].A)[0][0]))
            self.EntityPRList = sorted(self.EntityPRList, key=lambda classPR: classPR[1], reverse=True)
        print("直接实体进行PageRank得到的结果为")
        for index, _ in enumerate(self.EntityPRList):
            if _[0] != self.target:
                print("{}.{} PR值为{}".format(len(answerList) + 1, _[0], _[1]))
                answerList.append(_[0])
        for i in range(0, len(self.TopK)):
            print(
                "topK={}, diversity score={}, coverage score={}, good score={}".format(self.TopK[i], self.getDiversity(
                    answerList[0:self.TopK[i]]),
                                                                                       self.getCoverage(
                                                                                           answerList[0:self.TopK[i]]),
                                                                                       self.getGoodness(
                                                                                           answerList[0:self.TopK[i]])))
        print("--------------------------------------------------------------------------------------")

    def getEntityTFIDF(self):
        for node in self.entitySet:
            class_List = self.getClassByEntity(node)
            for class_ in class_List:
                tf = 1 / len(self.classEntityDict[class_])
                idf = math.log10(10000 / len(class_List))
                self.TF_IDF.append((node, class_, tf * idf))
        self.TF_IDF = sorted(self.TF_IDF, key=lambda TF_IDF: TF_IDF[2], reverse=True)
        print("通过TF,IDF得到的结果为")
        setTF = []
        for index, _ in enumerate(self.TF_IDF):
            if _[0] not in setTF and _[0] != self.target:
                print("{}. {} , {}, TF值为{}".format(len(setTF) + 1, _[0], _[1], _[2]))
                setTF.append(_[0])
        for i in range(0, len(self.TopK)):
            print(
                "topK={}, diversity score={}, coverage score={}, good score={}".format(self.TopK[i], self.getDiversity(
                    setTF[0:self.TopK[i]]),
                                                                                       self.getCoverage(
                                                                                           setTF[0:self.TopK[i]]),
                                                                                       self.getGoodness(
                                                                                           setTF[0:self.TopK[i]])))

        print("--------------------------------------------------------------------------------------")

    def getPopularityRank(self):
        for node in self.entitySet:
            self.EntityPopList.append((node, self.getPop(node)))
        self.EntityPopList = sorted(self.EntityPopList, key=lambda entity: entity[1])
        tempList = self.EntityPopList.copy()
        print("根据启发式原则得到的结果为")
        while tempList[-1][0] == self.target:
            tempList.pop()
        First = tempList.pop()
        chosenList = [First[0]]
        print("1. {} pop值为{}".format(First[0], First[1]))
        while len(tempList) != 0:
            discrepancyList = []
            for i in range(len(tempList) - 1, -1, -1):
                discrepancy = 0
                thisClassSet = set(self.getClassByEntity(tempList[i][0]))
                for chosen in chosenList:
                    chosenClassSet = set(self.getClassByEntity(chosen))
                    discrepancy += len(chosenClassSet | thisClassSet) - len(chosenClassSet & thisClassSet)
                discrepancyList.append(discrepancy)
            shouldIndex = len(discrepancyList) - 1 - discrepancyList.index(max(discrepancyList))
            if tempList[shouldIndex][0] == self.target:
                continue
            chosenList.append(tempList[shouldIndex][0])
            print("{}. {} pop值为{}".format(len(chosenList), tempList[shouldIndex][0], tempList[shouldIndex][1]))
            if len(chosenList) >= self.TopK[-1]:
                break
            del tempList[shouldIndex]
        for i in range(0, len(self.TopK)):
            print(
                "topK={}, diversity score={}, coverage score={}, good score={}".format(self.TopK[i], self.getDiversity(
                    chosenList[0:self.TopK[i]]),
                                                                                       self.getCoverage(
                                                                                           chosenList[0:self.TopK[i]]),
                                                                                       self.getGoodness(
                                                                                           chosenList[0:self.TopK[i]])))
        print("--------------------------------------------------------------------------------------")

    def getDiscrepancyEntity(self, e1, e2):
        c1 = set(self.getClassByEntity(e1))
        c2 = set(self.getClassByEntity(e2))
        return 1 - len(c1 & c2) / len(c1 | c2)

    def getDiversity(self, rankList):
        diversity = 0
        for i in range(0, len(rankList), 1):
            for j in range(i, len(rankList), 1):
                diversity += self.getDiscrepancyEntity(rankList[i], rankList[j])
        return 2 * diversity / len(rankList) / (len(rankList) - 1)

    def getCoverage(self, rankList):
        coverage = set()
        for i in rankList:
            class_List = self.getClassByEntity(i)
            for j in class_List:
                coverage.add(j)
        return len(coverage) / len(self.classSet)

    def getScore(self, rankList):
        assert self.score_Alpha + self.score_Beta + self.score_Omega == 1
        return self.score_Alpha * self.getDiversity(rankList) + \
               self.score_Beta * self.getCoverage(rankList) + self.score_Omega * self.getGoodness(rankList)

    def getGoodness(self, rankList):
        goodness = 0
        for i in rankList:
            goodness += self.getGood(i)
        return goodness / len(rankList)

    def casualExplore(self):
        self.getRelationOfClass()
        self.displayClassPR()
        self.displayEntityPR()
        self.getEntityTFIDF()
        self.getPopularityRank()


if __name__ == '__main__':
    target = "dbr:Tom_Hanks"
    limit = 2
    topK = [10, 50, 100]
    alpha_Relation = 0.5
    alpha_Wc = 0.3
    beta_Wc = 0.1
    omega_Wc = 0.6
    lamda = 0.5
    Fai = 0.5
    fai = 0.5
    d_PageRank = 0.85
    score_Alpha, score_Beta, score_Omega = 1 / 3, 1 / 3, 1 / 3
    casualExplore = CasualExplore(target, limit, topK, alpha_Relation, alpha_Wc, beta_Wc, omega_Wc, lamda, Fai, fai,
                                  d_PageRank, score_Alpha, score_Beta, score_Omega)
    casualExplore.casualExplore()
