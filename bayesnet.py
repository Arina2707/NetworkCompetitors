import math
import json
from pomegranate import *


class BayesNet:
    def __init__(self):
        self.network = BayesianNetwork()

    def make_scores(self):
        beliefs1 = self.network.predict_proba({})
        beliefs1 = map(str, beliefs1)
        z1 = "\n".join(
            "{}".format(belief) for state, belief in zip(self.network.states, beliefs1) if state.name == "Overall")
        z1 = json.loads(z1)
        result = round(z1.get("parameters")[0].get("yes"), 3)

        return result


class ProductBayesNet(BayesNet):
    def __init__(self):
        super(ProductBayesNet, self).__init__()

    @staticmethod
    def add_first_layer(e1, w1, e2, w2, e3, w3, dt, ap):
        product1_energy = DiscreteDistribution({'yes': e1, 'no': 1 - e1})
        product1_wl = DiscreteDistribution({'yes': w1, 'no': 1 - w1})

        product2_energy = DiscreteDistribution({'yes': e2, 'no': 1 - e2})
        product2_wl = DiscreteDistribution({'yes': w2, 'no': 1 - w2})

        product3_energy = DiscreteDistribution({'yes': e3, 'no': 1 - e3})
        product3_wl = DiscreteDistribution({'yes': w3, 'no': 1 - w3})
        avg_delivery_time = DiscreteDistribution({'yes': dt, 'no': 1 - dt})
        avg_price = DiscreteDistribution({'yes': ap, 'no': 1 - ap})

        return [product1_energy, product1_wl, product2_energy, product2_wl, product3_energy, product3_wl,
                avg_delivery_time, avg_price]

    @staticmethod
    def create_conditional_probs(l):
        product1 = ConditionalProbabilityTable(
            [['yes', 'no', 'yes', 0.67],
             ['no', 'yes', 'yes', 0.33],
             ['yes', 'yes','yes', 1.0],
             ['no', 'no','yes', 0.0],
             ['yes', 'no', 'no', 0.33],
             ['no', 'yes', 'no', 0.67],
             ['yes', 'yes', 'no', 0.0],
             ['no', 'no', 'no', 1.0]], [l[0], l[1]])

        product2 = ConditionalProbabilityTable(
            [['yes', 'no', 'yes', 0.33],
             ['no', 'yes', 'yes', 0.67],
             ['yes', 'yes','yes', 1.0],
             ['no', 'no','yes', 0.0],
             ['yes', 'no', 'no', 0.67],
             ['no', 'yes', 'no', 0.33],
             ['yes', 'yes', 'no', 0.0],
             ['no', 'no', 'no', 1.0]], [l[2], l[3]])

        technical_characteristics = ConditionalProbabilityTable(
            [['yes', 'no', 'yes', 0.33],
             ['no', 'yes', 'yes', 0.67],
             ['yes', 'yes','yes', 1.0],
             ['no', 'no','yes', 0.0],
             ['yes', 'no', 'no', 0.67],
             ['no', 'yes', 'no', 0.33],
             ['yes', 'yes', 'no', 0.0],
             ['no', 'no', 'no', 1.0]], [l[4], l[5]])

        product3 = ConditionalProbabilityTable(
            [['yes', 'no', 'yes', 'yes', 0.78],
             ['no', 'yes', 'yes', 'yes', 0.89],
             ['yes', 'yes', 'yes', 'yes', 1.0],
             ['no', 'no', 'yes', 'yes', 0.67],
             ['yes', 'no', 'no', 'yes', 0.11],
             ['no', 'yes', 'no', 'yes', 0.22],
             ['yes', 'yes', 'no', 'yes', 0.33],
             ['no', 'no', 'no', 'yes', 0.0],
             ['yes', 'no', 'yes', 'no', 0.22],
             ['no', 'yes', 'yes', 'no', 0.11],
             ['yes', 'yes', 'yes', 'no', 0.0],
             ['no', 'no', 'yes', 'no', 0.33],
             ['yes', 'no', 'no', 'no', 0.89],
             ['no', 'yes', 'no', 'no', 0.78],
             ['yes', 'yes', 'no', 'no', 0.67],
             ['no', 'no', 'no', 'no', 1.0]], [l[6],l[7],technical_characteristics])

        overall_product = ConditionalProbabilityTable(
            [['yes', 'no', 'yes', 'yes', 0.78],
             ['no', 'yes', 'yes', 'yes', 0.89],
             ['yes', 'yes', 'yes', 'yes', 1.0],
             ['no', 'no', 'yes', 'yes', 0.67],
             ['yes', 'no', 'no', 'yes', 0.11],
             ['no', 'yes', 'no', 'yes', 0.22],
             ['yes', 'yes', 'no', 'yes', 0.33],
             ['no', 'no', 'no', 'yes', 0.0],
             ['yes', 'no', 'yes', 'no', 0.22],
             ['no', 'yes', 'yes', 'no', 0.11],
             ['yes', 'yes', 'yes', 'no', 0.0],
             ['no', 'no', 'yes', 'no', 0.33],
             ['yes', 'no', 'no', 'no', 0.89],
             ['no', 'yes', 'no', 'no', 0.78],
             ['yes', 'yes', 'no', 'no', 0.67],
             ['no', 'no', 'no', 'no', 1.0]], [product1, product2, product3])

        return [product1, product2, technical_characteristics, product3, overall_product]

    def bake_network(self, e1, w1, e2, w2, e3, w3, dt, ap):
        nodes1 = self.add_first_layer(e1, w1, e2, w2, e3, w3, dt, ap)
        nodes2 = self.create_conditional_probs(nodes1)
        n = nodes1 + nodes2

        s1 = State(n[0], name="pr1_energy")
        s2 = State(n[1], name="pr1_wl")
        s3 = State(n[2], name="pr2_energy")
        s4 = State(n[3], name="pr2_wl")
        s5 = State(n[4], name="pr3_energy")
        s6 = State(n[5], name="pr3_wl")
        s7 = State(n[6], name="avg_delivery")
        s8 = State(n[7], name="avg_price")
        s9 = State(n[8], name="pr1")
        s10 = State(n[9], name="pr2")
        s11 = State(n[10], name="pr3_tech")
        s12 = State(n[11], name="pr3")
        s13 = State(n[13], name="Overall")

        self.network.add_states(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13)
        self.network.add_edge(s1, s9)
        self.network.add_edge(s2, s9)
        self.network.add_edge(s3, s10)
        self.network.add_edge(s4, s10)

        self.network.add_edge(s5, s11)
        self.network.add_edge(s6, s11)
        self.network.add_edge(s7, s12)
        self.network.add_edge(s8, s12)
        self.network.add_edge(s11, s12)

        self.network.add_edge(s9, s13)
        self.network.add_edge(s10, s13)
        self.network.add_edge(s12, s13)

        self.network.bake()


class TechBayesNet(BayesNet):
    def __init__(self):
        super(TechBayesNet, self).__init__()

    @staticmethod
    def add_first_layer(e1, w1, e2):
        sphere = DiscreteDistribution({'yes': e1, 'no': 1 - e1})
        tech = DiscreteDistribution({'yes': w1, 'no': 1 - w1})
        conf = DiscreteDistribution({'yes': e2, 'no': 1 - e2})

        return [sphere, tech, conf]

    @staticmethod
    def create_conditional_probs(l):
        applied_tech = ConditionalProbabilityTable(
            [['yes', 'no', 'yes', 0.33],
             ['no', 'yes', 'yes', 0.67],
             ['yes', 'yes','yes', 1.0],
             ['no', 'no','yes', 0.0],
             ['yes', 'no', 'no', 0.67],
             ['no', 'yes', 'no', 0.33],
             ['yes', 'yes', 'no', 0.0],
             ['no', 'no', 'no', 1.0]], [l[0], l[1]])

        tech_capability = ConditionalProbabilityTable(
            [['yes', 'no', 'yes', 0.33],
             ['no', 'yes', 'yes', 0.67],
             ['yes', 'yes','yes', 1.0],
             ['no', 'no','yes', 0.0],
             ['yes', 'no', 'no', 0.67],
             ['no', 'yes', 'no', 0.33],
             ['yes', 'yes', 'no', 0.0],
             ['no', 'no', 'no', 1.0]], [l[2], applied_tech])

        return [applied_tech, tech_capability]

    def bake_network(self, e1, w1, e2):
        nodes1 = self.add_first_layer(e1, w1, e2)
        nodes2 = self.create_conditional_probs(nodes1)
        n = nodes1 + nodes2

        s1 = State(n[0], name="sphere")
        s2 = State(n[1], name="possible_tech")
        s3 = State(n[2], name="conf")
        s4 = State(n[3], name="applied_tech")
        s5 = State(n[4], name="Overall")

        self.network.add_states(s1, s2, s3, s4, s5)
        self.network.add_edge(s1, s4)
        self.network.add_edge(s2, s4)
        self.network.add_edge(s3, s5)
        self.network.add_edge(s4, s5)

        self.network.bake()


class OrgBayesNet(BayesNet):
    def __init__(self):
        super(OrgBayesNet, self).__init__()

    @staticmethod
    def add_first_layer(e1, w1, e2):
        place = DiscreteDistribution({'yes': e1, 'no': 1 - e1})
        shares = DiscreteDistribution({'yes': w1, 'no': 1 - w1})
        cb = DiscreteDistribution({'yes': e2, 'no': 1 - e2})

        return [place, shares, cb]

    @staticmethod
    def create_conditional_probs(l):
        overall = ConditionalProbabilityTable(
            [['yes', 'no', 'yes', 'yes', 0.33],
             ['no', 'yes', 'yes', 'yes', 0.78],
             ['yes', 'yes', 'yes', 'yes', 1.0],
             ['no', 'no', 'yes', 'yes', 0.11],
             ['yes', 'no', 'no', 'yes', 0.22],
             ['no', 'yes', 'no', 'yes', 0.67],
             ['yes', 'yes', 'no', 'yes', 0.89],
             ['no', 'no', 'no', 'yes', 0.0],
             ['yes', 'no', 'yes', 'no', 0.67],
             ['no', 'yes', 'yes', 'no', 0.22],
             ['yes', 'yes', 'yes', 'no', 0.0],
             ['no', 'no', 'yes', 'no', 0.89],
             ['yes', 'no', 'no', 'no', 0.78],
             ['no', 'yes', 'no', 'no', 0.33],
             ['yes', 'yes', 'no', 'no', 0.11],
             ['no', 'no', 'no', 'no', 1.0]], [l[0], l[2], l[1]])

        return [overall]

    def bake_network(self, e1, w1, e2):
        nodes1 = self.add_first_layer(e1, w1, e2)
        nodes2 = self.create_conditional_probs(nodes1)
        n = nodes1 + nodes2

        s1 = State(n[0], name="place")
        s2 = State(n[1], name="shares")
        s3 = State(n[2], name="cb")
        s4 = State(n[3], name="Overall")

        self.network.add_states(s1, s2, s3, s4)
        self.network.add_edge(s1, s4)
        self.network.add_edge(s2, s4)
        self.network.add_edge(s3, s4)

        self.network.bake()


class CustomerBayesNet(BayesNet):
    def __init__(self):
        super(CustomerBayesNet, self).__init__()

    @staticmethod
    def add_first_layer(e1, w1, e2, w2, e3, w3, dt, ap):
        mensions = DiscreteDistribution({'yes': e1, 'no': 1 - e1})
        positiveness = DiscreteDistribution({'yes': w1, 'no': 1 - w1})
        subscribers = DiscreteDistribution({'yes': e2, 'no': 1 - e2})

        overall_rank = DiscreteDistribution({'yes': w2, 'no': 1 - w2})
        reach_rank = DiscreteDistribution({'yes': e3, 'no': 1 - e3})
        rank_per_mill = DiscreteDistribution({'yes': w3, 'no': 1 - w3})

        views_rank = DiscreteDistribution({'yes': dt, 'no': 1 - dt})
        views_per_user = DiscreteDistribution({'yes': ap, 'no': 1 - ap})

        return [mensions, positiveness, subscribers, overall_rank, reach_rank, rank_per_mill, views_rank, views_per_user]

    @staticmethod
    def create_conditional_probs(l):
        views = ConditionalProbabilityTable(
            [['yes', 'no', 'yes', 0.67],
             ['no', 'yes', 'yes', 0.33],
             ['yes', 'yes','yes', 1.0],
             ['no', 'no','yes', 0.0],
             ['yes', 'no', 'no', 0.33],
             ['no', 'yes', 'no', 0.67],
             ['yes', 'yes', 'no', 0.0],
             ['no', 'no', 'no', 1.0]], [l[6], l[7]])


        rank = ConditionalProbabilityTable(
            [['yes', 'no', 'yes', 'yes', 0.33],
             ['no', 'yes', 'yes', 'yes', 0.78],
             ['yes', 'yes', 'yes', 'yes', 1.0],
             ['no', 'no', 'yes', 'yes', 0.11],
             ['yes', 'no', 'no', 'yes', 0.22],
             ['no', 'yes', 'no', 'yes', 0.67],
             ['yes', 'yes', 'no', 'yes', 0.89],
             ['no', 'no', 'no', 'yes', 0.0],
             ['yes', 'no', 'yes', 'no', 0.67],
             ['no', 'yes', 'yes', 'no', 0.22],
             ['yes', 'yes', 'yes', 'no', 0.0],
             ['no', 'no', 'yes', 'no', 0.67],
             ['yes', 'no', 'no', 'no', 0.11],
             ['no', 'yes', 'no', 'no', 0.22],
             ['yes', 'yes', 'no', 'no', 0.33],
             ['no', 'no', 'no', 'no', 1.0]], [l[3],l[4],l[5]])

        reviews = ConditionalProbabilityTable(
            [['yes', 'no', 'yes', 'yes', 0.33],
             ['no', 'yes', 'yes', 'yes', 0.78],
             ['yes', 'yes', 'yes', 'yes', 1.0],
             ['no', 'no', 'yes', 'yes', 0.11],
             ['yes', 'no', 'no', 'yes', 0.22],
             ['no', 'yes', 'no', 'yes', 0.67],
             ['yes', 'yes', 'no', 'yes', 0.89],
             ['no', 'no', 'no', 'yes', 0.0],
             ['yes', 'no', 'yes', 'no', 0.67],
             ['no', 'yes', 'yes', 'no', 0.22],
             ['yes', 'yes', 'yes', 'no', 0.0],
             ['no', 'no', 'yes', 'no', 0.67],
             ['yes', 'no', 'no', 'no', 0.11],
             ['no', 'yes', 'no', 'no', 0.22],
             ['yes', 'yes', 'no', 'no', 0.33],
             ['no', 'no', 'no', 'no', 1.0]], [l[0],l[1],l[2]])

        website = ConditionalProbabilityTable(
            [['yes', 'no', 'yes', 0.33],
             ['no', 'yes', 'yes', 0.67],
             ['yes', 'yes','yes', 1.0],
             ['no', 'no','yes', 0.0],
             ['yes', 'no', 'no', 0.67],
             ['no', 'yes', 'no', 0.33],
             ['yes', 'yes', 'no', 0.0],
             ['no', 'no', 'no', 1.0]], [rank, views])

        overall = ConditionalProbabilityTable(
            [['yes', 'no', 'yes', 0.67],
             ['no', 'yes', 'yes', 0.33],
             ['yes', 'yes','yes', 1.0],
             ['no', 'no','yes', 0.0],
             ['yes', 'no', 'no', 0.33],
             ['no', 'yes', 'no', 0.67],
             ['yes', 'yes', 'no', 0.0],
             ['no', 'no', 'no', 1.0]], [reviews, website])

        return [views, rank, reviews, website, overall]

    def bake_network(self, e1, w1, e2, w2, e3, w3, dt, ap):
        nodes1 = self.add_first_layer(e1, w1, e2, w2, e3, w3, dt, ap)
        nodes2 = self.create_conditional_probs(nodes1)
        n = nodes1 + nodes2

        s1 = State(n[0], name="mensions")
        s2 = State(n[1], name="positiveness")
        s3 = State(n[2], name="subscribers")
        s4 = State(n[3], name="overall_rank")
        s5 = State(n[4], name="reach_rank")
        s6 = State(n[5], name="rank_per_mill")
        s7 = State(n[6], name="views_rank")
        s8 = State(n[7], name="views_per_user")
        s9 = State(n[8], name="views")
        s10 = State(n[9], name="rank")
        s11 = State(n[10], name="reviews")
        s12 = State(n[11], name="website")
        s13 = State(n[13], name="Overall")

        self.network.add_states(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13)
        self.network.add_edge(s1, s11)
        self.network.add_edge(s2, s11)
        self.network.add_edge(s3, s11)

        self.network.add_edge(s3, s9)
        self.network.add_edge(s4, s9)
        self.network.add_edge(s5, s9)
        self.network.add_edge(s6, s8)
        self.network.add_edge(s7, s8)

        self.network.add_edge(s8, s11)
        self.network.add_edge(s9, s11)
        self.network.add_edge(s10, s13)
        self.network.add_edge(s11, s13)

        self.network.bake()

