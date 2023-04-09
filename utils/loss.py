import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class OurContrastiveLoss(nn.Module):   # use only z(E,U)
    def __init__(self, w_size, mode, tau=0.04,masking_factor=1.):
        super(OurContrastiveLoss, self).__init__()
        self.tau = tau
        self.generated_sample = int(w_size * masking_factor)
        self.mode = mode

    def forward(self, E,U,B,C):

        if self.mode ==3:
            # normalize E, normalize B and calculate sigmoid of U,C
            nE = F.normalize(E, dim=1)
            nB = F.normalize(B, dim=1)
            sU = torch.sigmoid(U)
            sC = torch.sigmoid(C)

            # calculate covariance matrix of EE and EB, BE, BB

            cov1 = torch.mm(nE, nE.t().contiguous())
            # print('cov1')
            # print(cov1.shape)
            cov2 = torch.mm(nE, nB.t().contiguous())
            # print('cov2')
            # print(cov2.shape)

            cov3 = torch.mm(nB, nB.t().contiguous())
            # print('cov3')
            # print(cov3.shape)
            cov4 = torch.mm(nB, nE.t().contiguous())
            # print('cov4')
            # print(cov4.shape)



            # repeat sU into 2 parts
            sU1 = sU.repeat(1, nE.size(0))
            sU2 = sU.repeat(1, nB.size(0))

            #repeat sC into2 parts
            sC1 = sC.repeat(1, nE.size(0))
            sC2 = sC.repeat(1,nB.size(0))

            # calculate simEE
            simEE = torch.exp((cov1 * sU1) / self.tau)  # exp(sim(zi,zj)tau(zi))

            # calculate simEB
            simEB = torch.exp((cov2 * sU2) / self.tau)  # exp(sim(zi,bj)tau(zi))

            # calculate simBE
            simBE = torch.exp((cov4 * sC1) / self.tau)  # exp(sim(bi,zj)tau(bi))

            # calculate simBB
            simBB = torch.exp((cov3 * sC2) / self.tau)  # exp(sim(bi,bj)tau(zi))

            # remove diagonal elements in simEE
            simEE = simEE.flatten()[1:].view(E.shape[0] - 1, E.shape[0] + 1)[:, :-1].reshape(E.shape[0], E.shape[
                0] - 1)  # (aWS,aWS-1) remove diagonal elements

            # remove diagonal elements in simBB
            simBB_neg = simBB.flatten()[1:].view(B.shape[0] - 1, B.shape[0] + 1)[:, :-1].reshape(B.shape[0], B.shape[
                0] - 1)  # (aWS,aWS-1) remove diagonal elements

            # #gather disimilar elements of simEE
            EE_negativelist = torch.zeros((len(E), len(E) - 1 - (self.generated_sample - 1)))
            #
            for rowindex in range(len(EE_negativelist)):
                EE_positivelist = torch.arange(
                    start=(math.floor(rowindex / self.generated_sample) * self.generated_sample), end=int(math.floor(
                        rowindex / self.generated_sample) * self.generated_sample + self.generated_sample - 1), step=1)
                nlist = torch.arange(0, len(E) - 1, 1)
                nlist = torch.cat([nlist[0: int(EE_positivelist[0])], nlist[int(EE_positivelist[-1]) + 1:]])
                EE_negativelist[rowindex] = nlist

            EE_negativelist = EE_negativelist.type(torch.int64).cuda()
            # print(EE_negativelist)

            simEE_neg = torch.gather(simEE, 1, EE_negativelist)


            # gather similar and disimilar elements of simEB
            EB_positivelist = torch.zeros((len(E), 1))
            # print('positivelist index')
            # print(positivelist.shape)

            EB_negativelist = torch.zeros((len(E), len(B) - 1))
            # print('negativelist index')
            # print(negativelist.shape)

            #

            for rowindex1 in range(len(EB_positivelist)):
                EB_positivelist[rowindex1][0] = math.floor(rowindex1 / self.generated_sample)
                nlist1 = torch.arange(0, len(B), 1)
                nlist1 = torch.cat(
                    [nlist1[0: int(EB_positivelist[rowindex1][0])], nlist1[int(EB_positivelist[rowindex1][0]) + 1:]])
                EB_negativelist[rowindex1] = nlist1

            EB_positivelist = EB_positivelist.type(torch.int64).cuda()
            EB_negativelist = EB_negativelist.type(torch.int64).cuda()

            simEB_pos = torch.gather(simEB, 1, EB_positivelist)
            simEB_neg = torch.gather(simEB, 1, EB_negativelist)


            # gather similar and disimilar elements of simBE
            BE_positivelist = torch.zeros((len(B), self.generated_sample))
            # print('positivelist index')
            # print(positivelist.shape)

            BE_negativelist = torch.zeros((len(B), len(E) - self.generated_sample))
            # print('negativelist index')
            # print(negativelist.shape)

            #

            for rowindex2 in range(len(BE_positivelist)):
                BE_positivelist[rowindex2] = torch.arange(start=rowindex2* self.generated_sample, end=rowindex2* self.generated_sample+ self.generated_sample,step=1 )
                nlist2 = torch.arange(0, len(E), 1)
                nlist2 = torch.cat(
                    [nlist2[0: int(BE_positivelist[rowindex2][0])], nlist2[int(BE_positivelist[rowindex2][-1]) + 1:]])
                BE_negativelist[rowindex2] = nlist2

            BE_positivelist = BE_positivelist.type(torch.int64).cuda()
            BE_negativelist = BE_negativelist.type(torch.int64).cuda()

            simBE_pos = torch.gather(simBE, 1, BE_positivelist)
            simBE_neg = torch.gather(simBE, 1, BE_negativelist)

            '---'
            simEE_neg = torch.sum(simEE_neg, dim=1)
            simEE_neg = torch.reshape(simEE_neg, shape=(len(simEE_neg), 1))

            simEB_neg = torch.sum(simEB_neg,dim=1)
            simEB_neg = torch.reshape(simEB_neg,shape=(len(simEB_neg),1))

            total_neg = torch.add(simEB_neg,simEE_neg)
            total_neg = torch.add(total_neg, simEB_pos)  # denominator   (#include negative pairs and itself)

            pairloss = torch.div(simEB_pos, total_neg)
            pairloss = -torch.log(pairloss)

            simBE_neg = torch.sum(simBE_neg,dim=1)
            simBE_neg = torch.reshape(simBE_neg, shape=(len(simBE_neg), 1))

            simBB_neg = torch.sum(simBB_neg,dim=1)
            simBB_neg = torch.reshape(simBB_neg, shape=(len(simBB_neg), 1))
            total_negBE = torch.add(simBE_neg,simBB_neg)
            total_negBE = total_negBE.repeat(1, self.generated_sample)
            total_negBE = torch.add(total_negBE, simBE_pos)  # denominator   (#include negative pairs and itself)

            pairloss2 = torch.div(simBE_pos, total_negBE)
            pairloss2 = -torch.log(pairloss2)

            return ((torch.sum(pairloss)/ torch.numel(pairloss)) + (torch.sum(pairloss2)/ torch.numel(pairloss2)))/2
            pass




        elif self.mode == 1:
            # print(E.shape, U.shape, B.shape)

            #normalize E, normalize B and calculate sigmoid of U
            nE = F.normalize(E, dim=1)
            nB = F.normalize(B, dim=1)
            sU = torch.sigmoid(U)

            #calculate covariance matrix of EE and EB

            cov1 = torch.mm(nE,nE.t().contiguous())
            # print('cov1')
            # print(cov1.shape)
            cov2 = torch.mm(nE,nB.t().contiguous())
            # print('cov2')
            # print(cov2.shape)

            #repeat sU into 2 parts
            sU1 = sU.repeat(1, nE.size(0))
            sU2 = sU.repeat(1, nB.size(0))
            # print('sU1')
            # print(sU1.shape)
            # print('sU2')
            # print(sU2.shape)

            # calculate simEE
            simEE = torch.exp((cov1 * sU1) / self.tau)      # exp(sim(zi,zj)tau(zi))

            # calculate simEB
            simEB = torch.exp((cov2 * sU2) / self.tau)      # exp(sim(zi,bj)tau(zi))

            # remove diagonal elements in simEE
            simEE = simEE.flatten()[1:].view(E.shape[0] - 1, E.shape[0] + 1)[:, :-1].reshape(E.shape[0], E.shape[0] - 1)  # (aWS,aWS-1) remove diagonal elements

            # #gather disimilar elements of simEE
            EE_negativelist = torch.zeros((len(E), len(E) - 1 - (self.generated_sample - 1)))
            #
            for rowindex in range(len(EE_negativelist)):
                EE_positivelist= torch.arange(start=(math.floor(rowindex / self.generated_sample) * self.generated_sample), end=int(math.floor(rowindex / self.generated_sample) * self.generated_sample + self.generated_sample - 1), step=1)
                nlist = torch.arange(0, len(E) - 1, 1)
                nlist = torch.cat([nlist[0: int(EE_positivelist[0])], nlist[int(EE_positivelist[-1]) + 1:]])
                EE_negativelist[rowindex] = nlist

            EE_negativelist = EE_negativelist.type(torch.int64).cuda()
            # print(EE_negativelist)

            simEE_neg = torch.gather(simEE, 1, EE_negativelist)


            #gather similar and disimilar elements of simEU
            EB_positivelist = torch.zeros((len(E), 1))
            # print('positivelist index')
            # print(positivelist.shape)

            EB_negativelist = torch.zeros((len(E), len(B) - 1 ))
            # print('negativelist index')
            # print(negativelist.shape)

            #

            for rowindex1 in range(len(EB_positivelist)):
                EB_positivelist[rowindex1][0] = math.floor(rowindex1 / self.generated_sample)
                nlist1 = torch.arange(0, len(B), 1)
                nlist1 = torch.cat(
                    [nlist1[0: int(EB_positivelist[rowindex1][0])], nlist1[int(EB_positivelist[rowindex1][0])+1:]])
                EB_negativelist[rowindex1] = nlist1

            EB_positivelist = EB_positivelist.type(torch.int64).cuda()
            EB_negativelist = EB_negativelist.type(torch.int64).cuda()

            simEB_pos = torch.gather(simEB, 1, EB_positivelist)
            simEB_neg = torch.gather(simEB, 1, EB_negativelist)

            # print('simEE_neg_' + str(simEE_neg.shape))
            # print('simEB_pos_' + str(simEB_pos.shape))
            # print('simEB_neg_' + str(simEB_neg.shape))

            simEE_neg = torch.sum(simEE_neg, dim=1)
            simEE_neg = torch.reshape(simEE_neg, shape=(len(simEE_neg), 1))

            simEB_neg = torch.sum(simEB_neg,dim=1)
            simEB_neg = torch.reshape(simEB_neg,shape=(len(simEB_neg),1))

            total_neg = torch.add(simEB_neg,simEE_neg)
            total_neg = torch.add(total_neg, simEB_pos)  # denominator   (#include negative pairs and itself)

            pairloss = torch.div(simEB_pos, total_neg)
            pairloss = -torch.log(pairloss)

            return torch.sum(pairloss)/ torch.numel(pairloss)


            pass
            # EUmul = torch.mm(E, U.t())  # (aWS, S)
            #
            # nE = torch.linalg.norm(E, dim=1, ord=2)
            # nE = torch.reshape(nE, shape=(len(nE), 1))  # (aWS,1)
            #
            # nU = torch.linalg.norm(U, dim=1, ord=2)
            # nU = torch.reshape(nU, shape=(len(nU), 1))  # (S,1)
            #
            # nEnUmul = torch.mul(nE, nU.t()) # (aWS,S)
            # zEU = torch.exp(torch.div(EUmul, torch.mul(nEnUmul,self.tau)))
            #
            # # gather positive elements
            # zEUsim = torch.zeros((len(E), 1)).cuda()
            # for rowindex in range(len(zEUsim)):
            #     zEUsim[rowindex][0] = zEU[rowindex][math.floor(rowindex/self.generated_sample)]
            #
            #
            # EEmul = torch.mm(E, E.t())  # (aWS, aWS)
            # nEnEmul = torch.mm(nE, nE.t())  # (aWS,aWS)
            # nEnEmul = torch.mul(nEnEmul, self.tau)
            # zEE = torch.div(EEmul, nEnEmul)
            # zEE = torch.exp(zEE)
            # zEE = zEE.flatten()[1:].view(E.shape[0] - 1, E.shape[0] + 1)[:, :-1].reshape(E.shape[0], E.shape[
            #     0] - 1)  # (aWS,aWS-1) remove diagonal elements
            #
            # #gather disimilar elements of zEE
            #
            # # gather positive elements and negative elements
            # negativelist = torch.zeros((len(E), len(E) - 1 - (self.generated_sample - 1)))
            #
            # for rowindex in range(len(negativelist)):
            #     positivelist= torch.arange(start=(math.floor(rowindex / self.generated_sample) * self.generated_sample), end=int(math.floor(rowindex / self.generated_sample) * self.generated_sample + self.generated_sample - 1), step=1)
            #     nlist = torch.arange(0, len(E) - 1, 1)
            #     nlist = torch.cat([nlist[0: int(positivelist[0])], nlist[int(positivelist[-1]) + 1:]])
            #     negativelist[rowindex] = nlist
            #
            # negativelist = negativelist.type(torch.int64).cuda()
            # zEEdisim = torch.gather(zEE, 1, negativelist)
            #
            #
            # #concat zEU with zEEdis
            # zdisim = torch.cat((zEEdisim, zEU), dim=1)
            #
            # zdisim = torch.sum(zdisim, dim=1)       #denom
            # zdisim = torch.reshape(zdisim, shape=(len(zdisim),1))   #denom
            #
            # #calculate loss
            # pairloss = torch.div(zEUsim, zdisim)
            # pairloss = -torch.log(pairloss)
            #
            # return torch.sum(pairloss)/ torch.numel(pairloss)
            #
            # pass


        elif self.mode ==2:
            # print(E.shape)
            # print(U.shape)

            #normalize features and apply sigmoid on uncertainty
            nE = F.normalize(E, dim=1)
            sU = torch.sigmoid(U)

            cov = torch.mm(nE, nE.t().contiguous())      #(aWS, aWS)
            # print('cov')
            # print(cov.shape)

            sU = sU.repeat(1, nE.size(0))
            # print('sU')
            # print(sU.shape)

            sim = torch.exp((cov * sU) / self.tau)      # exp(sim(zi,zj)tau(zi))
            # print('sim')
            # print(sim.shape)

            #remove diagonal elements
            sim = sim.flatten()[1:].view(E.shape[0]-1, E.shape[0]+1)[:,:-1].reshape(E.shape[0], E.shape[0]-1)     #(aWS,aWS-1) remove diagonal elements
            # print('sim gaian')
            # print(sim.shape)

            # #gather positive elements and negative elements
            positivelist = torch.zeros((len(E), self.generated_sample-1))
            # print('positivelist index')
            # print(positivelist.shape)

            negativelist = torch.zeros((len(E), len(E) -1 - (self.generated_sample-1)))
            # print('negativelist index')
            # print(negativelist.shape)

            #
            for rowindex in range(len(positivelist)):
                positivelist[rowindex] = torch.arange(start=(math.floor(rowindex / self.generated_sample) * self.generated_sample),
                                                  end=int(math.floor(rowindex / self.generated_sample) * self.generated_sample + self.generated_sample-1), step=1)
                nlist = torch.arange(0, len(E) - 1, 1)
                nlist = torch.cat([nlist[0: int(positivelist[rowindex][0])], nlist[int(positivelist[rowindex][-1]) + 1:]])
                negativelist[rowindex] = nlist

            positivelist = positivelist.type(torch.int64).cuda()
            negativelist = negativelist.type(torch.int64).cuda()
            pos_sim = torch.gather(sim, 1, positivelist)     #nominator
            neg_sim = torch.gather(sim,1, negativelist)
            neg_sim = torch.sum(neg_sim,dim=1)
            neg_sim = torch.reshape(neg_sim, shape=(len(neg_sim),1))
            neg_sim = neg_sim.repeat(1, pos_sim.shape[1])
            neg_sim = torch.add(neg_sim,pos_sim)     # denominator   (#include negative pairs and itself)

            pairloss = torch.div(pos_sim, neg_sim)
            pairloss = -torch.log(pairloss)

            return torch.sum(pairloss)/ torch.numel(pairloss)




class ReconstructionLoss(nn.Module):
    def __init__(self,window_size):
        super(ReconstructionLoss, self).__init__()
        self.window_size = window_size


    def forward(self, output, target):

        # a = torch.sub(output, target)   #element-wise subtraction
        # print(a.shape)
        # a = torch.linalg.norm(a, dim=1)
        # a = torch.sum(a, dim=1)
        # a = torch.div(a, self.window_size)
        return torch.sum(torch.div(torch.sum(torch.linalg.norm(torch.sub(output, target), dim=1), dim=1), self.window_size))


