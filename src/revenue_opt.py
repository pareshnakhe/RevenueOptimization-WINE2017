import numpy as np
import math
from random import uniform
import matplotlib.pyplot as plt

numItems = 5
rho = 0.5
crnt_prices = np.zeros(numItems)
for i in range(0, numItems):
    crnt_prices[i] = uniform(0.1, 2.0)

class CES_util():

    def __init__(self, numItems, rho, crnt_prices):
        self.Income = 1
        self.rho = rho
        self.numItems = numItems
        self.E = 1.0 / (1 - self.rho)

        # New addition
        self.crnt_dmd = np.zeros(numItems)
        self.crnt_prices = crnt_prices
        # self.crnt_prices = np.zeros(numItems)
        # for i in range(0, self.numItems):
        #     self.crnt_prices[i] = uniform(0.1, 2.0)
        # print self.crnt_prices

        # placeholders of length = # of sellers
        self.crnt_supply = np.ones(numItems)
        self.crnt_dmd = self.compute_demand()
        self.crnt_revenue = np.zeros(numItems)

    # https://wwz.unibas.ch/fileadmin/wwz/redaktion/witheo/personen/georg/HS12/amic_prob_1_12.pdf
    def compute_demand(self):
        temp_dmd = np.zeros(self.numItems)
        temp = 0
        temp_index = self.rho / (self.rho - 1.0)
        for i in range(0, self.numItems):
            #print "price[%d]: %f" % (i, self.crnt_prices[i])
            temp += self.crnt_prices[i]**temp_index
            # print temp
            if math.isnan(temp):
                print("NaN error")
                exit(1)

        for i in range(0, self.numItems):
            temp_dmd[i] = (self.crnt_prices[i]**(temp_index - 1)) / temp

        self.set_crnt_dmd(temp_dmd)
        return temp_dmd

    # r_i(p) = min( p_i * x_i(p), p_i * w_i)
    def compute_revenue(self):
        revenue_list = [min(self.crnt_prices[i]*self.crnt_dmd[i], self.crnt_prices[i]*self.crnt_supply[i]) for i in range(0, self.numItems)]
        self.crnt_revenue = np.asarray(revenue_list)

    def set_crnt_dmd(self, temp_dmd):
        self.crnt_dmd = temp_dmd

    def get_crnt_dmd(self):
        return self.crnt_dmd

    def set_crnt_price(self, seller_index, price):
        self.crnt_prices[seller_index] = price

    def get_crnt_price(self, seller_index):
        return self.crnt_prices[seller_index]

    def get_crnt_revenue(self, seller_index):
        return self.crnt_revenue[seller_index]

    def compute_surrogate_loss(self):
        dmd = self.compute_demand()
        return math.pow(math.log(dmd[0]), 2)

    # This method returns the updated prices
    # typ is used to decide the update rule to apply
    def update_price(self, typ, round_no, previous_excess_dmd):

        crnt_excess_dmd = self.crnt_dmd - self.crnt_supply
        # typ = 1 is the Cole's update method. Constant step size
        if typ == 1:
            # print self.crnt_prices
            for i in range(0, self.numItems):
                self.crnt_prices[i] *= (1 + math.pow(2.0*self.E, -1) * (float(self.crnt_dmd[i] - self.crnt_supply[i])/float(self.crnt_supply[i])))
                if self.crnt_prices[i] == 0:
                    print "price getting too small/large"
                    exit(1)
        # typ = 2 is Cole's method with decreasing step size
        elif typ == 2:
            # print self.crnt_prices
            for i in range(0, self.numItems):
                self.crnt_prices[i] *= (1 + math.pow(round_no, -0.50) * math.pow(2.0 * self.E, -1) * (float(self.crnt_dmd[i] - self.crnt_supply[i]) / float(self.crnt_supply[i])))
                if self.crnt_prices[i] == 0:
                    print "price getting too small/large"
                    exit(1)
        # typ = 3 is the OGD (Zinkevich)
        elif typ == 3:
            for i in range(0, self.numItems):
                if crnt_excess_dmd[i] >= 0.0:
                    self.crnt_prices[i] *= (1.0 + math.pow(round_no, -0.50) * math.pow(1.0 * self.E, -1))
                else:
                    self.crnt_prices[i] *= (1.0 - math.pow(round_no, -0.50) * math.pow(1.0 * self.E, -1))
        # typ = 4 optimistic mirror descent
        elif typ == 4:
            for i in range(0, self.numItems):
                self.crnt_prices[i] *= (1 + math.pow(round_no, -0.250) * math.pow(2.0 * self.E, -1) * (float(2.0 * crnt_excess_dmd[i] - previous_excess_dmd[i])))
        # typ = 5. Only experimental
        elif typ == 5:
            for i in range(0, self.numItems):
                if crnt_excess_dmd[i] >= 0.0:
                    self.crnt_prices[i] *= (1.0 + math.pow(round_no, -0.750) * math.pow(1.0 * self.E, -1))
                else:
                    self.crnt_prices[i] *= (1.0 - math.pow(round_no, -0.750) * math.pow(1.0 * self.E, -1))
        # typ = 5. Only experimental
        elif typ == 6:
            for i in range(0, self.numItems):
                if crnt_excess_dmd[i] >= 0.0:
                    self.crnt_prices[i] *= (1.0 + math.pow(round_no, -0.501) * math.pow(1.0 * self.E, -1))
                else:
                    self.crnt_prices[i] *= (1.0 - math.pow(round_no, -0.501) * math.pow(1.0 * self.E, -1))
        # typ = 0: update other prices as is the OGD (Zinkevich), keep price of item "0" fixed
        # This is a naive strategy for player 0. Increment by 0.10 if excess dmd positive, else decrement.
        elif typ == 0:
            temp = self.get_crnt_price(0)
            for i in range(0, self.numItems):
                if crnt_excess_dmd[i] >= 0.0:
                    self.crnt_prices[i] *= (1.0 + math.pow(round_no, -0.50) * math.pow(1.0 * self.E, -1))
                else:
                    self.crnt_prices[i] *= (1.0 - math.pow(round_no, -0.50) * math.pow(1.0 * self.E, -1))
            if crnt_excess_dmd[0] >= 0.0:
                self.set_crnt_price(0, temp + 0.01)
            else:
                # decrement only if result non-negative.
                if temp - 0.01 > 0:
                    self.set_crnt_price(0, temp - 0.01)

    def update_supply(self):
        for i in range(0, self.numItems):
            self.crnt_supply[i] = uniform(1.0, 1.5)

    def get_crnt_supply(self):
        return self.crnt_supply


# This method executes the Cole's update method.
def cole_method(typ):
    # This parameter is used to decide the type of update to be applied
    # 1 - Cole's update method
    T = 700
    # Excess demand is a matrix of size numItems * T. Holds the entire history
    excess_dmd = np.zeros((numItems, T))
    #maintains revenue values for each round for seller 0
    revenue_0 = list()
    temp_excess_dmd = np.zeros(numItems)

    for t in range(0, T):

        # Get demand vector for chosen prices
        dmd = engine.compute_demand()

        engine.compute_revenue()
        revenue_0.append(engine.get_crnt_revenue(0))

        supply = engine.get_crnt_supply()
        for i in range(numItems):
            excess_dmd[i][t] = dmd[i] - supply[i]
        # Update prices
        engine.update_price(typ, t+1, temp_excess_dmd)
        temp_excess_dmd = dmd - supply

        # Update supplies
        # engine.update_supply()

    # Plot the excess demand for seller 0
    # print excess_dmd[0]
    # base = np.zeros(T)
    # plt.plot(excess_dmd[0], 'r--', base, 'b--')
    # plt.show()

    return excess_dmd[0], np.asarray(revenue_0)
    #return np.asarray(revenue_0)


def revenue_plot():
    log_revenue = list()
    revenue = list()
    price_list = [x / 100.0 for x in range(1, 1000)]

    for price in price_list:
        engine.set_crnt_price(0, price)
        engine.compute_demand()
        engine.compute_revenue()
        # revenue.append(engine.get_crnt_revenue(0))
        log_revenue.append(math.log(engine.get_crnt_revenue(0)))
        revenue.append(engine.get_crnt_revenue(0))
        price += 0.10

    log_price = [math.log(x) for x in price_list]
    #plt.plot(np.asarray(price_list), np.asarray(revenue), 'r--')
    plt.plot(np.asarray(log_price), np.asarray(log_revenue), 'r--')
    plt.show()


def surrogate_loss_plot():
    numItems = 5
    crnt_prices = np.zeros(numItems)
    for i in range(0, numItems):
        crnt_prices[i] = uniform(0.1, 2.0)

    engine1 = CES_util(numItems, 0.5, crnt_prices)
    engine2 = CES_util(numItems, 0.9, crnt_prices)
    surrogate_loss1 = list()     # for seller 0
    surrogate_loss2 = list()
    price_list = [x / 10000.0 for x in range(1, 80000)]

    for price in price_list:
        engine1.set_crnt_price(0, price)
        engine2.set_crnt_price(0, price)
        #engine.compute_demand()
        temp1 = engine1.compute_surrogate_loss()
        temp2 = engine2.compute_surrogate_loss()
        surrogate_loss1.append(temp1)
        surrogate_loss2.append(temp2)
        price += 0.10

    log_price = [math.log(x) for x in price_list]
    #plt.plot(np.asarray(price_list), np.asarray(revenue), 'r--')
    #plt.plot(np.asarray(log_price), np.asarray(surrogate_loss), 'r--')
    plt.plot(np.asarray(log_price), np.asarray(surrogate_loss1), '-r', linewidth=2.5, label='E=2')
    plt.plot(np.asarray(log_price), np.asarray(surrogate_loss2), '-g', linewidth=2.5, label='E=10')
    plt.legend(loc='upper left')
    plt.ylabel('Surrogate loss', fontsize=15)
    plt.xlabel('log price', fontsize=15)
    #plt.show()
    plt.savefig("E2-10")


np.seterr(all='warn', over='raise')
#engine = CES_util(numItems)
surrogate_loss_plot()
exit(0)

# typ=0 : Naive approach
# typ=1 : OGD on Surrogate loss, constant step size
# typ=2 : OGD on Surrogate loss, decreasing step size
# typ=3 : OGD on modified revenue

engine = CES_util(numItems)
ex3, rev3 = cole_method(typ=3)
engine = CES_util(numItems)
ex0, rev0 = cole_method(typ=0)
engine = CES_util(numItems)
ex2, rev2 = cole_method(typ=2)
engine = CES_util(numItems)
ex1, rev1 = cole_method(typ=1)


# Part 1: OGD on modified revenue vs naive approach

# Plot for Excess demand
plt.plot(ex3, '-g', linewidth=2.0, label='OGD:Revenue')
plt.plot(ex0, '-m', linewidth=2.0, label='Naive Approach')
plt.legend(loc='lower right')
plt.ylabel('Excess demand', fontsize=14)
plt.xlabel('rounds', fontsize=14)
plt.savefig("E-" + str(int(engine.E)) + "part1-1")

# Plot for Revenue
plt.clf()
plt.plot(rev3, '-g', linewidth=2.0, label='OGD:Revenue')
plt.plot(rev0, '-m', linewidth=2.0, label='Naive Approach')
plt.legend(loc='lower right')
plt.ylabel('Revenue', fontsize=14)
plt.xlabel('rounds', fontsize=14)
plt.savefig("E-" + str(int(engine.E)) + "part1-2")


# Part 2: OGD with dec step size on surrogate loss vs OGD on modified revenue


# Plot for Excess demand
plt.clf()
plt.plot(ex2, '-b', linewidth=2.5, label='OGD:Surrogate Loss')
plt.plot(ex3, '-g', linewidth=2.5, label='OGD:Revenue')
plt.legend(loc='lower right')
plt.ylabel('Excess demand', fontsize=14)
plt.xlabel('rounds', fontsize=14)
plt.savefig("E-" + str(int(engine.E)) + "part2-1")

# Plot for Revenue
plt.clf()
plt.plot(rev2, '-b', linewidth=2.5, label='OGD:Surrogate Loss')
plt.plot(rev3, '-g', linewidth=2.5, label='OGD:Revenue')
plt.legend(loc='lower right')
plt.ylabel('Revenue', fontsize=14)
plt.xlabel('rounds', fontsize=14)
plt.savefig("E-" + str(int(engine.E)) + "part2-2")

# Part 3: OGD:surrogate with constant step size vs OGD:surrogate with dec step size

# Plot for Excess demand
plt.clf()
plt.plot(ex1, '-b', linewidth=3.0, label='OGD:constant step')
plt.plot(ex2, '-r', linewidth=3.0, label='OGD:dec step')
plt.legend(loc='lower right')
plt.ylabel('Excess demand', fontsize=14)
plt.xlabel('rounds', fontsize=14)
plt.savefig("E-" + str(int(engine.E)) + "part3-1")

# Plot for Revenue
plt.clf()
plt.plot(rev1, '-b', linewidth=3.0, label='OGD:constant step')
plt.plot(rev2, '-r', linewidth=3.0, label='OGD:dec step')
plt.legend(loc='lower right')
plt.ylabel('Revenue', fontsize=14)
plt.xlabel('rounds', fontsize=14)
plt.savefig("E-" + str(int(engine.E)) + "part3-2")
