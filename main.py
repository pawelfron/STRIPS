from stripsForwardPlanner import STRIPS_domain, Strips, Forward_STRIPS
from searchMPP import SearcherMPP
from stripsProblem import Planning_problem, boolean

from typing import Set, Dict
import os

at = lambda b, a: f"{b}_is_at_{a}"
minerals = lambda a: f"there_are_minerals_on_{a}"
collected_minerals = lambda b: f"{b}_collected_minerals"
empty = lambda a: f"{a}_is_empty"
building = lambda a: f"there_is_a_building_on_{a}"
scv = lambda b: f"{b}_is_an_scv"
location = lambda a: f"{a}_is_a_location"

marine = lambda a: f"there_is_a_marine_at_{a}"
tank = lambda a: f"there_is_a_tank_at_{a}"
wraith = lambda a: f"there_is_a_wraith_at_{a}"
battlecruiser = lambda a: f"there_is_a_battlecruiser_at_{a}"
depot = lambda a: f"there_is_a_depot_at_{a}"
barracks = lambda a: f"there_are_barracks_at_{a}"
factory = lambda a: f"there_is_a_factory_at_{a}"
starport = lambda a: f"there_is_a_starport_at_{a}"
fusion_core = lambda a: f"there_is_a_fusion_core_at_{a}"

move = lambda b, a1, a2: f"move_{b}_from_{a1}_to_{a2}"
collect_minerals = lambda b, a: f"{b}_collect_minerals_at_{a}"
build_supply_depot = lambda b, a: f"{b}_build_supply_depot_at_{a}"
build_barracks = lambda b, a1, a2: f"{b}_build_barracks_at_{a1}_depot_at_{a2}"
build_factory = lambda b, a1, a2: f"{b}_build_factory_at_{a1}_barracks_at_{a2}"
build_starport = lambda b, a1, a2: f"{b}_build_starport_at_{a1}_factory_at_{a2}"
build_fusion_core = lambda b, a1, a2: f"{b}_build_fusion_core_at_{a1}_starport_at_{a2}"
train_marine = lambda b, a: f"{b}_train_marine_at_{a}"
train_tank = lambda b, a: f"{b}_train_tank_at_{a}"
train_wraith = lambda b, a: f"{b}_train_wraith_at_{a}"
train_battlecruiser = lambda b, a1, a2: f"{b}_train_battlecruiser_at_{a1}_fusion_core_at_{a2}"

def create_StarCraft_domain(builders: Set[str], areas: Set[str]) -> STRIPS_domain:
    features = {at(b, a): boolean for b in builders for a in areas}
    features |= {minerals(a): boolean for a in areas}
    features |= {collected_minerals(b): boolean for b in builders}
    features |= {empty(a): boolean for a in areas}
    features |= {building(a): boolean for a in areas}
    features |= {scv(b): boolean for b in builders}
    features |= {location(a): boolean for a in areas}

    features |= {marine(a): boolean for a in areas}
    features |= {tank(a): boolean for a in areas}
    features |= {wraith(a): boolean for a in areas}
    features |= {battlecruiser(a): boolean for a in areas}
    features |= {depot(a): boolean for a in areas}
    features |= {barracks(a): boolean for a in areas}
    features |= {factory(a): boolean for a in areas}
    features |= {starport(a): boolean for a in areas}
    features |= {fusion_core(a): boolean for a in areas}

    actions = {Strips(move(b, a1, a2), 
                      {scv(b): True, location(a1): True, location(a2): True, at(b, a1): True, at(b, a2): False}, 
                      {at(b, a2): True, at(b, a1): False}) 
               for b in builders 
               for a1 in areas 
               for a2 in areas 
               if a1 != a2}
    actions |= {Strips(collect_minerals(b, a), 
                       {scv(b): True, location(a): True, minerals(a): True, empty(a): False, at(b, a): True}, 
                       {empty(a): True, collected_minerals(b): True})
                for b in builders
                for a in areas}
    actions |= {Strips(build_supply_depot(b, a),
                       {scv(b): True, location(a): True, collected_minerals(b): True, at(b, a): True, building(a): False, minerals(a): False}, 
                       {building(a): True, depot(a): True, collected_minerals(b): False})
                for b in builders
                for a in areas}
    actions |= {Strips(build_barracks(b, a1, a2),
                       {scv(b): True, location(a1): True, location(a2): True, depot(a2): True, collected_minerals(b): True, at(b, a1): True, building(a1): False, minerals(a1): False}, 
                       {building(a1): True, barracks(a1): True, collected_minerals(b): False})
                for b in builders
                for a1 in areas
                for a2 in areas
                if a1 != a2}
    actions |= {Strips(build_factory(b, a1, a2),
                       {scv(b): True, location(a1): True, location(a2): True, barracks(a2): True, collected_minerals(b): True, at(b, a1): True, building(a1): False, minerals(a1): False}, 
                       {building(a1): True, factory(a1): True, collected_minerals(b): False})
                for b in builders
                for a1 in areas
                for a2 in areas
                if a1 != a2}
    actions |= {Strips(build_starport(b, a1, a2),
                       {scv(b): True, location(a1): True, location(a2): True, factory(a2): True, collected_minerals(b): True, at(b, a1): True, building(a1): False, minerals(a1): False}, 
                       {building(a1): True, starport(a1): True, collected_minerals(b): False})
                for b in builders
                for a1 in areas
                for a2 in areas
                if a1 != a2}
    actions |= {Strips(build_fusion_core(b, a1, a2),
                       {scv(b): True, location(a1): True, location(a2): True, starport(a2): True, collected_minerals(b): True, at(b, a1): True, building(a1): False, minerals(a1): False}, 
                       {building(a1): True, fusion_core(a1): True, collected_minerals(b): False})
                for b in builders
                for a1 in areas
                for a2 in areas
                if a1 != a2}
    actions |= {Strips(train_marine(b, a), 
                       {collected_minerals(b): True, barracks(a): True}, 
                       {marine(a): True, collected_minerals(b): False})
                for b in builders
                for a in areas}
    actions |= {Strips(train_tank(b, a), 
                       {collected_minerals(b): True, factory(a): True}, 
                       {tank(a): True, collected_minerals(b): False})
                for b in builders
                for a in areas}
    actions |= {Strips(train_wraith(b, a), 
                       {collected_minerals(b): True, starport(a): True}, 
                       {wraith(a): True, collected_minerals(b): False})
                for b in builders
                for a in areas}
    actions |= {Strips(train_battlecruiser(b, a1, a2), 
                       {collected_minerals(b): True, starport(a1): True, fusion_core(a2): True}, 
                       {battlecruiser(a1): True, collected_minerals(b): False})
                for b in builders
                for a1 in areas
                for a2 in areas
                if a1 != a2}
    print(f"number of features: {len(features)}; number of actions: {len(actions)}")
    return STRIPS_domain(features, actions)

def create_StarCraft_problem(domain: STRIPS_domain, initial_state: Dict[str, bool], goal: Dict[str, bool]) -> Planning_problem:
    features = domain.feature_domain_dict.keys()
    full_initial_state = {feature: False for feature in features}
    for key, value in initial_state.items():
        full_initial_state[key] = value
    return Planning_problem(domain, full_initial_state, goal)

builders = {"scv"}
areas = {"sectorA", "sectorB", "mineralFieldA", "mineralFieldB"}
domain = create_StarCraft_domain(builders, areas)
problem = create_StarCraft_problem(domain, 
                                   {scv("scv"): True, location("sectorA"): True, location("sectorB"): True, location("mineralFieldA"): True, location("mineralFieldB"): True, minerals("mineralFieldA"): True, minerals("mineralFieldB"): True, at("scv", "sectorA"): True}, 
                                   {barracks("sectorA"): True})
searcher = SearcherMPP(Forward_STRIPS(problem))
solution = searcher.search()

if not os.path.exists("results"):
    os.makedirs("results")

with open("results/out.txt", "w") as file:
    result = ""
    solution_str = solution.__repr__().split("\n")
    for i, line in enumerate(solution_str[1:]):
        start = line.find("--")
        end = line.find("-->")
        result += f"{i + 1}. {line[(start + 2):end]}\n"
    file.write(result)