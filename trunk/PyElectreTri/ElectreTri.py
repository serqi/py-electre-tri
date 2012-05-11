import random
import cPickle as pickle
from pulp import GLPK, LpProblem, LpVariable, LpMaximize, LpBinary

class SemanticError(Exception):
	def __init__(self,value):
		self.value = value

	def __str__(self):
		return self.value

class Criterium:
	def __init__(self, name, weight = 1, pref = 0, indif = 0, veto = 0):

		self.name = name
		self.weight = weight
		self.pref = pref
		self.indif = indif
		self.veto = veto

		self.check_constraints()
			
	def check_constraints(self):
		
		self.check_tresholds()
		self.check_weight()

	def check_tresholds(self):
		"""Check the preference/indifference/veto tresholds for """

		if self.veto:
			if (self.indif > self.pref) or (self.pref > self.veto):
				raise SemanticError, "there is an inconsistance regarding the preference/indifference/veto tresholds of criterium [" + self.name + "]. Must be like indif<=pref<=veto." 			
		
		if self.indif > self.pref:
			raise SemanticError, "there is an inconsistance regarding the preference/indifference tresholds [" + self.name + "]. Must be like indif<=pref."

	def check_weight(self):

		if self.weight == 0:
			raise SemanticError, "Weight may not be equal to zero for criterium [" + self.name + "]."
			

class Alternative:
	def __init__(self, name, category = None):

		self.name = name
		self.category = category
		self.refCategory = category


class LimitProfile(Alternative):
	pass

class Category:
	def __init__(self,rank , name = None, lp_inf = None, lp_sup = None):
		self.name = name
		self.lp_inf = lp_inf
		self.lp_sup = lp_sup
		self.rank = rank #relative rank

class PerformanceTable:	
	def __init__(self, alts, crits):

		self.create_performance_table(alts,crits)
		self.alts = alts
		self.crits = crits

	def create_performance_table(self, alts, crits):
		self.performance_table = {}
		for alt in alts:
			self.performance_table[alt] = {}
			for crit in crits:
				self.performance_table[alt][crit] = None

	def set_perf(self, alt, crit, value):

		self.performance_table[alt][crit] = value

	def get_perf(self, alt, crit):

		return self.performance_table[alt][crit]

	def randomize(self, minimum = 0, maximum = 1, seed = None):
		"""will create a randomized performance table by initializing
		 every cell to a random float number between "minimum" and
		"maximum" (between 0 and 1 by default). It is also possible to define
		a seed (by providing the parameter seed).
		"""
	
		random.seed(seed)
		for alt in self.alts:
			for crit in self.crits:
				self.performance_table[alt][crit] = random.uniform(minimum,maximum)

	def __str__(self):

		ptstr=""
		for alt in self.alts:
			ptstr += alt.name + " : "
			for crit in self.crits:
				ptstr += "(" + crit.name + " -> "
				ptstr += str(self[alt][crit]) + ") "
			ptstr += "\n" 
		return ptstr

	def __getitem__(self, alt):

		return self.performance_table[alt]

	def __setitem__(self, alt, value):

		self.performance_table[alt] = value

	def __delitem__(self, alt):
		
		del self.performance_table[alt]
		self.alts.remove(alt)

	def __iter__(self):

		self.index = len(self.alts)
		return self

	def next(self):

		if self.index == 0:			
			raise StopIteration		
		self.index = self.index - 1
		return self.alts[self.index]

	def append_alternative(self, alt, *args):
		""" Append a new alternative line in the performance table.
		It is possible to pass the values directly with additional arguments
		"""
	
		self.alts.append(alt)

		values = list(args)

		self.performance_table[alt] = {}
		for crit in self.crits:
			if len(values):
				self.performance_table[alt][crit] = values.pop(0)
			else:
				self.performance_table[alt][crit] = None
				
	def set_reference(self):
		"""set the Accuracy to 100% for the current assignation for alternatives to the categories.
		"""

		for alt in self.alts:
			alt.refCategory = alt.category

	def get_accuracy(self):
		"""return the accuracy of the trained model in prct. i.e. the percentage of alternatives 
		that were well classified.
		First, run the learn method, then a solver (optimistic or pessimistic), then this method.
		"""
		alts = filter(lambda a: a.__class__.__name__ != "LimitProfile", self.alts)

		n_alts = len(alts)
		n_good = 0.0

		for alt in alts:
			if alt.refCategory is alt.category:
				n_good += 1

		return n_good / n_alts * 100
	

#--------------------------ELECTRE TRI methods------------------------

class ElectreTriSimple:
	def __init__(self, crits, categories, cutting_threshold = 0.5):

		self.cutting_threshold = cutting_threshold
		self.crits = crits
		self.limit_profiles = list(set([cat.lp_inf for cat in categories if cat.lp_inf is not None] + [cat.lp_sup for cat in categories if cat.lp_sup is not None]))
		self.categories = categories

		#self.check_data_consistence()
		self.normalize_weights() #the total weight sum must be equal to one

	def normalize_weights(self):
		totalWeight = 0.0		
		for crit in self.crits:
			totalWeight += crit.weight
		for crit in self.crits:
			crit.weight /= totalWeight

	def surclass(self, alt1, alt2, performance_table):	
		"""return true if alt1 surclasses alt2."""	
		sum = 0.0
		for crit in self.crits:
			if performance_table[alt1][crit] >= performance_table[alt2][crit]:
				sum += crit.weight
		return sum >= self.cutting_threshold

	def solve_pessimistic(self, performance_table):
		#sort the categories by ascending rank order:
		self.categories.sort(key = lambda c : c.rank)

		#Go through all the alternatives:
		alts = performance_table.alts	#list of all the alternatives, including the limit profiles
		for alt in alts:
			if alt.__class__.__name__ != "LimitProfile": #ignore the limit profiles
				for cat in self.categories:
					if cat.lp_inf is None \
					or (self.surclass(alt, cat.lp_inf, performance_table)and not self.surclass(cat.lp_inf, alt, performance_table)):
						alt.category = cat
						break					

	def solve_optimistic(self, performance_table):
		#sort the categories by descending rank order:
		self.categories.sort(key = lambda c : c.rank, reverse = True)	
		
		#Go through all the alternatives:
		alts = performance_table.alts	#list of all the alternatives, including the limit profiles
		for alt in alts:
			if alt.__class__.__name__ != "LimitProfile": #ignore the limit profiles
				for cat in self.categories:
					if cat.lp_sup is None \
					or (self.surclass(cat.lp_sup, alt, performance_table) and not self.surclass(alt, cat.lp_sup, performance_table)):
						alt.category = cat
						break

	def learn_two_cat(self, performance_table):
		"""Learn parameters for two categories and two training sets of alternatives."""		
		if len(self.categories) != 2:
			raise SemanticError, "The learnTwoCat() method requires exactly two categories."

		prob = LpProblem("twoCat", LpMaximize)

		alts = filter(lambda a: a.__class__.__name__ != "LimitProfile", performance_table.alts)
		crits = self.crits

		alts_name = [alt.name for alt in alts if alt.__class__.__name__ != "LimitProfile"]
		crits_name = [crit.name for crit in crits]

		self.categories.sort(key=lambda c: c.rank)

		alts1 = [alt for alt in alts if alt.category == self.categories[0]]
		alts2 = [alt for alt in alts if alt.category == self.categories[1]]

		#small float number
		epsilon = 0.00001

		#variables (v: variable, d: dict of variables)
		v_lambda = LpVariable("lambda", lowBound=0.5, upBound=1)
		v_alpha = LpVariable("alpha", lowBound=0)
		d_x = LpVariable.dicts("x", alts_name, lowBound=0)
		d_y = LpVariable.dicts("y", alts_name, lowBound=0)
		d_p = LpVariable.dicts("p", crits_name, lowBound=0, upBound=1)
		d_gb = LpVariable.dicts("gb", crits_name, lowBound=0, upBound=1)
		d_delta = LpVariable.dicts("delta", \
			[alt.name + crit.name for alt in alts for crit in crits]\
			, cat=LpBinary)
		d_c = LpVariable.dicts("c", \
			[alt.name + crit.name for alt in alts for crit in crits]\
			, lowBound=0, upBound=1)

		#maximize
		prob += v_alpha

		#constraints
		for alt in alts2:
			prob += sum(d_c[alt.name + crit.name] for crit in crits) + d_x[alt.name] == v_lambda
		for alt in alts1:
			prob += sum(d_c[alt.name + crit.name] for crit in crits) == v_lambda + d_y[alt.name]

		for alt in alts:
			prob += v_alpha <= d_x[alt.name]
			prob += v_alpha <= d_y[alt.name]
			prob += d_x[alt.name] >= epsilon
			
			for crit in crits:
				prob += d_c[alt.name + crit.name] <= d_p[crit.name]
				prob += d_c[alt.name + crit.name] >= d_delta[alt.name + crit.name] + d_p[crit.name] - 1
				prob += d_c[alt.name + crit.name] <= d_delta[alt.name + crit.name]
				prob += d_delta[alt.name + crit.name] >= performance_table[alt][crit] - d_gb[crit.name] + epsilon
				prob += d_delta[alt.name + crit.name] <= performance_table[alt][crit] - d_gb[crit.name] + 1

		prob += sum(d_p[crit.name] for crit in crits) == 1

		#solver
		GLPK().solve(prob)

		#prob.writeLP("SpamClassification.lp")
		#status = prob.solve()


		#update parameters
		self.cutting_threshold = v_lambda.value()

		for crit in crits:
			crit.weight = d_p[crit.name].value()

		for crit in crits:
			performance_table[self.limit_profiles[0]][crit] = d_gb[crit.name].value()
			
	def learn_two_cat2(self, performance_table):
		"""This version of learnTwoCat checks whether or not an alternative is to be kept
		in the learning process. This method is also faster than the previous version
		""" 
		
		if len(self.categories) != 2:
			raise SemanticError, "The learnTwoCat() method requires exactly two categories."

		prob = LpProblem("twoCat2", LpMaximize)
		
		alts = filter(lambda a: a.__class__.__name__ != "LimitProfile", performance_table.alts)
		crits = self.crits

		alts_name = [alt.name for alt in alts]
		crits_name = [crit.name for crit in crits]

		self.categories.sort(key=lambda c: c.rank)

		alts1 = [alt for alt in alts if alt.category == self.categories[0]]
		alts2 = [alt for alt in alts if alt.category == self.categories[1]]

		#small float number
		epsilon = 0.0001

		#variables (v: variable, d: dict of variables)
		v_lambda = LpVariable("lambda", lowBound=0.5, upBound=1)
		d_gamma = LpVariable.dicts("gamma", alts_name, cat=LpBinary)
		d_p = LpVariable.dicts("p", crits_name, lowBound=0, upBound=1)
		d_gb = LpVariable.dicts("gb", crits_name, lowBound=0, upBound=1)
		d_delta = LpVariable.dicts("delta", \
			[alt.name + crit.name for alt in alts for crit in crits]\
			, cat=LpBinary)
		d_c = LpVariable.dicts("c", \
			[alt.name + crit.name for alt in alts for crit in crits]\
			, lowBound=0, upBound=1)

		#maximize
		prob += sum(d_gamma[alt.name] for alt in alts)

		#constraints
		for alt in alts2:
			prob += sum(d_c[alt.name + crit.name] for crit in crits) + epsilon <= v_lambda + 2 * (1 - d_gamma[alt.name])
		for alt in alts1:
			prob += sum(d_c[alt.name + crit.name] for crit in crits) >= v_lambda - 2 * (1 - d_gamma[alt.name])
			
		for alt in alts:
			for crit in crits:
				prob += d_c[alt.name + crit.name] <= d_p[crit.name]
				prob += d_c[alt.name + crit.name] <= d_delta[alt.name + crit.name]
				prob += d_c[alt.name + crit.name] >= d_delta[alt.name + crit.name] + d_p[crit.name] - 1
				prob += d_delta[alt.name + crit.name] >= performance_table[alt][crit] - d_gb[crit.name] + epsilon
				prob += d_delta[alt.name + crit.name] <= performance_table[alt][crit] - d_gb[crit.name] + 1
				
		prob += sum(d_p[crit.name] for crit in crits)  == 1

		#solver
		GLPK().solve(prob)

		#update parameters
		self.cutting_threshold = v_lambda.value()

		for crit in crits:
			crit.weight = d_p[crit.name].value()

		for crit in crits:
			performance_table[self.limit_profiles[0]][crit] = d_gb[crit.name].value()
		 
		self.ignoredAlternatives = []
		for alt in alts:
			if d_gamma[alt.name].value() == 0:
				self.ignoredAlternatives.append(alt)

	def learn(self, performance_table):
		"""Learn parameters"""

		prob = LpProblem("Learn", LpMaximize)

		alts = filter(lambda a: a.__class__.__name__ != "LimitProfile", performance_table.alts)
		crits = self.crits

		alts_name = [alt.name for alt in alts]
		crits_name = [crit.name for crit in crits]
		
		categories = self.categories
		 
		categories.sort(key=lambda c: c.rank, reverse=True)	
		
		categoriesUp = list(categories)
		firstCat = categoriesUp.pop(0)
		
		categoriesDown = list(categories)
		lastCat = categoriesDown.pop()
		
		categories0 = list(categories)
		categories0.insert(0, Category(rank = (categories[0].rank + 1), name = "fake")) #add a fake category on the first position
				
		alternativesByCat = {}
		for cat in categories:
			alternativesByCat[cat] = [alt for alt in alts if alt.category == cat]
			
		#small float number
		epsilon = 0.000001

		#variables (v: variable, d: dict of variables)
		v_lambda = LpVariable("lambda", lowBound=0.5, upBound=1)
		v_alpha = LpVariable("alpha", lowBound=0)
		d_x = LpVariable.dicts("x", alts_name, lowBound=0)
		d_y = LpVariable.dicts("y", alts_name, lowBound=0)
		d_p = LpVariable.dicts("p", crits_name, lowBound=0, upBound=1)
		d_gb = LpVariable.dicts("gb", \
								[crit.name + cat.name for crit in crits for cat in categories0], \
								lowBound=0, \
								upBound=1)
		d_deltaInf = LpVariable.dicts("deltaInf", \
									  [alt.name + crit.name for alt in alts for crit in crits], \
									  cat=LpBinary)
		d_deltaSup = LpVariable.dicts("deltaSup", \
									  [alt.name + crit.name for alt in alts for crit in crits], \
									  cat=LpBinary)
		d_cInf = LpVariable.dicts("cInf", \
								  [alt.name + crit.name for alt in alts for crit in crits], \
								  lowBound=0, upBound=1)
		d_cSup = LpVariable.dicts("cSup", \
								  [alt.name + crit.name for alt in alts for crit in crits], \
								  lowBound=0, upBound=1)

		#maximize
		prob += v_alpha

		#constraints
		for crit in crits:
			prob += d_gb[crit.name + "fake"] == 0
			prob += d_gb[crit.name + lastCat.name] == 1
		

		for cat in categoriesDown:
			for alt in alternativesByCat[cat]:
				prob += sum(d_cSup[alt.name + crit.name] for crit in crits) + d_x[alt.name] == v_lambda
		for cat in categoriesUp:
			for alt in alternativesByCat[cat]:
				prob += sum(d_cInf[alt.name + crit.name] for crit in crits) == v_lambda + d_y[alt.name]
				
		for alt in alts:
			prob += v_alpha <= d_x[alt.name]
			prob += v_alpha <= d_y[alt.name]
			prob += d_x[alt.name] >= epsilon
			
			for crit in crits:
				prob += d_cInf[alt.name + crit.name] <= d_p[crit.name]
				prob += d_cSup[alt.name + crit.name] <= d_p[crit.name]
				prob += d_cInf[alt.name + crit.name] <= d_deltaInf[alt.name + crit.name]
				prob += d_cSup[alt.name + crit.name] <= d_deltaSup[alt.name + crit.name]
				prob += d_cInf[alt.name + crit.name] >= d_deltaInf[alt.name + crit.name] + d_p[crit.name] - 1
				prob += d_cSup[alt.name + crit.name] >= d_deltaSup[alt.name + crit.name] + d_p[crit.name] - 1
				
		prev_cat_name = "fake"
		for cat in categories:
			for alt in alternativesByCat[cat]:
				for crit in crits:
					prob += d_deltaInf[alt.name + crit.name] >= performance_table[alt][crit] - d_gb[crit.name + prev_cat_name] + epsilon
					prob += d_deltaSup[alt.name + crit.name] >= performance_table[alt][crit] - d_gb[crit.name + cat.name] + epsilon
					prob += d_deltaInf[alt.name + crit.name] <= performance_table[alt][crit] - d_gb[crit.name + prev_cat_name] + 1
					prob += d_deltaSup[alt.name + crit.name] <= performance_table[alt][crit] - d_gb[crit.name + cat.name] + 1
			prev_cat_name = cat.name

		prev_cat_name = firstCat.name
		for cat in categoriesUp:
			for crit in crits:
				prob += d_gb[crit.name + cat.name] >= d_gb[crit.name + prev_cat_name]
			prev_cat_name = cat.name
			
		prob += sum(d_p[crit.name] for crit in crits) == 1
		
		print prob
		
		#solver
		GLPK().solve(prob)

		#prob.writeLP("SpamClassification.lp")
		#status = prob.solve()


		#update parameters
		self.cutting_threshold = v_lambda.value()

		for crit in crits:
			crit.weight = d_p[crit.name].value()
			
		for cat in categoriesDown:
			for crit in crits:
				performance_table[cat.lp_sup][crit] = d_gb[crit.name + cat.name].value()
				
	def learn2(self, performance_table):
		"""This version of learn checks whether or not an alternative is to be kept
		in the learning process. This method is also faster than the previous version
		""" 

		prob = LpProblem("Learn", LpMaximize)

		alts = filter(lambda a: a.__class__.__name__ != "LimitProfile", performance_table.alts)
		crits = self.crits

		alts_name = [alt.name for alt in alts]
		crits_name = [crit.name for crit in crits]
		
		categories = self.categories
		
		categories.sort(key=lambda c: c.rank, reverse=True)	
		
		categoriesUp = list(categories)
		firstCat = categoriesUp.pop(0)
		
		categoriesDown = list(categories)
		lastCat = categoriesDown.pop()
		
		categories0 = list(categories)
		categories0.insert(0, Category(rank = (categories[0].rank + 1), name = "fake")) #add a fake category on the first position
				
		alternativesByCat = {}
		for cat in categories:
			alternativesByCat[cat] = [alt for alt in alts if alt.category == cat]
			
		#small float number
		epsilon = 0.001

		#variables (v: variable, d: dict of variables)
		v_lambda = LpVariable("lambda", lowBound=0.5, upBound=1)
		d_gamma = LpVariable.dicts("gamma", alts_name, cat=LpBinary)
		d_p = LpVariable.dicts("p", crits_name, lowBound=0, upBound=1)
		d_gb = LpVariable.dicts("gb", \
								[crit.name + cat.name for crit in crits for cat in categories0], \
								lowBound=0, \
								upBound=1)
		d_deltaInf = LpVariable.dicts("deltaInf", \
									  [alt.name + crit.name for alt in alts for crit in crits], \
									  cat=LpBinary)
		d_deltaSup = LpVariable.dicts("deltaSup", \
									  [alt.name + crit.name for alt in alts for crit in crits], \
									  cat=LpBinary)
		d_cInf = LpVariable.dicts("cInf", \
								  [alt.name + crit.name for alt in alts for crit in crits], \
								  lowBound=0, upBound=1)
		d_cSup = LpVariable.dicts("cSup", \
								  [alt.name + crit.name for alt in alts for crit in crits], \
								  lowBound=0, upBound=1)

		#maximize
		prob += sum(d_gamma[alt.name] for alt in alts)

		#constraints		
		for crit in crits:
			prob += d_gb[crit.name + "fake"] == 0
			prob += d_gb[crit.name + lastCat.name] == 1
					
		for cat in categoriesDown:
			for alt in alternativesByCat[cat]:
				tmp =  alt.name + crit.name #fixed a weird bug with pulp
				prob += sum(d_cSup[tmp] for crit in crits) + epsilon <= v_lambda + 2 * (1 - d_gamma[alt.name])
		for cat in categoriesUp:
			for alt in alternativesByCat[cat]:
				tmp =  alt.name + crit.name #fixed a weird bug with pulp
				prob += sum(d_cInf[tmp] for crit in crits) >= v_lambda - 2 * (1 - d_gamma[alt.name])					

		for alt in alts:	   
			for crit in crits:
				prob += d_cInf[alt.name + crit.name] <= d_p[crit.name]
				prob += d_cSup[alt.name + crit.name] <= d_p[crit.name]
				prob += d_cInf[alt.name + crit.name] <= d_deltaInf[alt.name + crit.name]
				prob += d_cSup[alt.name + crit.name] <= d_deltaSup[alt.name + crit.name]
				prob += d_cInf[alt.name + crit.name] >= d_deltaInf[alt.name + crit.name] + d_p[crit.name] - 1
				prob += d_cSup[alt.name + crit.name] >= d_deltaSup[alt.name + crit.name] + d_p[crit.name] - 1
				
		prev_cat_name = "fake"
		for cat in categories:
			for alt in alternativesByCat[cat]:
				for crit in crits:
					prob += d_deltaInf[alt.name + crit.name] >= performance_table[alt][crit] - d_gb[crit.name + prev_cat_name] + epsilon
					prob += d_deltaSup[alt.name + crit.name] >= performance_table[alt][crit] - d_gb[crit.name + cat.name] + epsilon
					prob += d_deltaInf[alt.name + crit.name] <= performance_table[alt][crit] - d_gb[crit.name + prev_cat_name] + 1
					prob += d_deltaSup[alt.name + crit.name] <= performance_table[alt][crit] - d_gb[crit.name + cat.name] + 1
			prev_cat_name = cat.name

		prev_cat_name = firstCat.name
		for cat in categoriesUp:
			for crit in crits:
				prob += d_gb[crit.name + cat.name] >= d_gb[crit.name + prev_cat_name]
			prev_cat_name = cat.name
			
		prob += sum(d_p[crit.name] for crit in crits) == 1
		
		print prob
		
		#solver
		GLPK().solve(prob)
		
		#update parameters
		self.cutting_threshold = v_lambda.value()

		for crit in crits:
			crit.weight = d_p[crit.name].value()
			
		for cat in categoriesDown:
			for crit in crits:
				performance_table[cat.lp_sup][crit] = d_gb[crit.name + cat.name].value()		
				
		self.ignoredAlternatives = []
		for alt in alts:
			if d_gamma[alt.name].value() == 0:
				self.ignoredAlternatives.append(alt)

	def check_data_consistence(self):
		"""Semantic check for the attributes of this class."""

		self.check_categories()
		#self.check_performance_table()
		self.check_limit_profiles()		
		self.check_cutting_threshold()

	def check_performance_table(self):
		"""Check if the performance table is filled with not None values, raise a SemanticError otherwise."""

		for alt in self.alts:
			for crit in self.crits:
				if self.performance_table[alt][crit] is None:
					raise SemanticError, "None value in performance Table for alternative [" + alt.name + "] and criterium [" + crit.name + "]."

	def check_cutting_threshold(self):
		"""Check if the cutting treshold must be set between 0.5 and 1, raise a SemanticError exception otherwise."""

		if not (0.5 <= self.cutting_threshold <= 1):
			raise SemanticError, "The cutting treshold must be set between 0.5 and 1"

	def check_categories(self):
		
		#check duplicated ranks:
		for cat1 in self.categories:
			for cat2 in self.categories:
				if cat1 is not cat2 and cat1.rank == cat2.rank:
					raise SemanticError, "Duplicated rank in the categories. A category's rank must be unique."

	def check_limit_profiles(self):	
		"""Check that category A has a limit profile in common with category B if A and B
		are subsequent categories.
		Check also if each sup limit profile surclasses the inf limit profile oof the same category.
		"""
		
		#pre-sort the categories by ascending rank
		orderedCategories = sorted(self.categories, key = lambda c: c.rank)

		#subsequent limit profiles:
		prev_cat = None
		for cat in orderedCategories:
			if (prev_cat is not None) and (cat.lp_sup != prev_cat.lp_inf):
				raise SemanticError, "Limit profile sup for category [" + cat.name + "] does not match limit profile inf for category [" + prev_cat.name + "]."
			prev_cat = cat

		#ordered limit profiles:
		for cat in orderedCategories:
			if (cat.lp_sup is not None) and (cat.lp_inf is not None) and (not self.surclass(cat.lp_sup, cat.lp_inf)):
				raise SemanticError, "The sup limit profile [" + cat.lp_sup.name + "] does not surclass the inf limit profile [" + cat.lp_inf.name + "] for the category [" + cat.name + "]." 

	def __str__(self):
		etsstr = "cutting threshold -> " + str(self.cutting_threshold)
		etsstr += "\ncategories -> " + str([cat.name for cat in self.categories])
		etsstr += "\ncriteria -> " + str([crit.name + "(" + str(crit.weight) + ")" for crit in self.crits])
		return etsstr
	
	
	def save(self, filename):
		"""Save the Electre Tri Model in a file on the disk.
		"""
		pickle.dump(self, open(filename, "wb"))
		
	def load(self, filename):
		"""Load an Electre Tri Model in a file on the disk.
		"""
		return pickle.load(open(filename))


#-----------------------------module tester--------------------------

if __name__ == "__main__":
	a1 = Alternative("a1")
	a2 = Alternative("a2")
	a3 = Alternative("a3")
	a4 = Alternative("a4")

	c1 = Criterium("c1")
	c2 = Criterium("c2")
	c3 = Criterium("c3")
	c4 = Criterium("c4")

	lp1 = LimitProfile("lp1")

	class1 = Category(1, name = "class1", lp_inf = lp1)
	class2 = Category(2, name = "class2", lp_sup = lp1)

	alist = [a1,a2,a3,a4,lp1]
	clist = [c1,c2,c3,c4]
	cllist = [class1,class2]

	pt = PerformanceTable(alist, clist)

	pt.randomize()

	print "\nPerformance table : \n"

	print pt

	ct = 0.7	#cutting threshold
	
	ets = ElectreTriSimple(clist, cllist, ct)
	
	print ets

	print "\nPessimistic :\n"

	ets.solve_pessimistic(pt)
	
	
	print [alt.category.name for alt in pt.alts if alt.__class__.__name__ != "LimitProfile"]

	print "\nOptimistic :\n"
	
	ets.solve_optimistic(pt)
	print [alt.category.name for alt in pt.alts if alt.__class__.__name__ != "LimitProfile"]
	pt.set_reference()
	

	ets.learn_two_cat(pt)

	print "c1 -> ", c1.weight
	print "c2 -> ", c2.weight
	print "c3 -> ", c3.weight
	print "c4 -> ", c4.weight

	print pt

	ets.solve_optimistic(pt)
	print ets
	print pt.get_accuracy()
	
	ets.save("spam_ham.ets")
	
	print ets
	
	print ets.load("spam_ham.ets")
	
	