:- use_module('metagol').

metagol:max_clauses(1).


% Facts from the knowledge base

action_giving_flowers(boy3, girl3).
action_receiving_flowers(girl3, boy3).
expression_happy(girl3).
object_flowers(bouquet3).
property_colorful(bouquet3).
clothing_neutral_or_warm(boy3).
clothing_soft_or_feminine(girl3).
environment_includes(symbol3).
symbol_affection(symbol3).

action_giving_flowers(boy4, girl4).
action_receiving_flowers(girl4, boy4).
expression_happy(girl4).
object_flowers(bouquet4).
property_colorful(bouquet4).
clothing_neutral_or_warm(boy4).
clothing_soft_or_feminine(girl4).

action_giving_flowers(boy0, girl0).
action_receiving_flowers(girl0, boy0).
expression_happy(girl0).
object_flowers(bouquet0).
property_colorful(bouquet0).
clothing_neutral_or_warm(boy0).
clothing_soft_or_feminine(girl0).
environment_includes(symbol0).
symbol_affection(symbol0).

action_giving_flowers(boy1, girl1).
action_receiving_flowers(girl1, boy1).
expression_happy(girl1).
object_flowers(bouquet1).
property_colorful(bouquet1).
clothing_neutral_or_warm(boy1).
clothing_soft_or_feminine(girl1).
environment_includes(symbol1).
symbol_affection(symbol1).

action_giving_flowers(boy2, girl2).
action_receiving_flowers(girl2, boy2).
expression_happy(girl2).
object_flowers(bouquet2).
property_colorful(bouquet2).
clothing_neutral_or_warm(boy2).
clothing_soft_or_feminine(girl2).
environment_includes(symbol2).
symbol_affection(symbol2).


% Body predicates

body_pred(action_giving_flowers/2).
body_pred(action_receiving_flowers/2).
body_pred(expression_happy/1).
body_pred(object_flowers/1).
body_pred(property_colorful/1).
body_pred(clothing_neutral_or_warm/1).
body_pred(clothing_soft_or_feminine/1).
body_pred(environment_includes/1).
body_pred(symbol_affection/1).



% Metarules

metarule([P,Q],[P,A,B],[[Q,A,B]]).
metarule([P,Q,R],[P,A,B],[[Q,A,B],[R,B,A]]).
%metarule([P,Q,R],[P,A,B],[[Q,C],[R,C]]).
metarule([P,Q,R],[P,A,B],[[Q,A],[R,B]]).
metarule([P,Q,R],[P,A,B],[[Q,B],[R,B]]).



% Learning Task

a :- Pos = [f(boy0, girl0), f(boy1, girl1), f(boy2, girl2)],

     Neg = [f(boy3, girl3), f(boy4, girl4), f(boy5, girl5)],

     learn(Pos, Neg).
