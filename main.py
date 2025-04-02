from aipython.stripsForwardPlanner import STRIPS_domain, Strips, Forward_STRIPS
from aipython.searchMPP import SearcherMPP
from aipython.stripsProblem import Planning_problem, boolean

from typing import Set, Dict, List, Tuple
import os
from time import time

at = lambda b, a: f"{b} is at {a}"
minerals = lambda a: f"there are minerals on {a}"
collected_minerals = lambda b: f"{b} collected_minerals"
empty = lambda a: f"{a} is empty"
building = lambda a: f"there is a building on {a}"
scv = lambda b: f"{b} is_an_scv"
location = lambda a: f"{a} is_a_location"

marine = lambda a: f"there is a marine at {a}"
tank = lambda a: f"there is a tank at {a}"
wraith = lambda a: f"there is a wraith at {a}"
battlecruiser = lambda a: f"there is a battlecruiser at {a}"
depot = lambda a: f"there is a depot at {a}"
barracks = lambda a: f"there are barracks at {a}"
factory = lambda a: f"there is a factory at {a}"
starport = lambda a: f"there is a starport at {a}"
fusion_core = lambda a: f"there is a fusion core at {a}"

move = lambda b, a1, a2: f"move {b} from {a1} to {a2}"
collect_minerals = lambda b, a: f"{b} collect minerals at {a}"
build_supply_depot = lambda b, a: f"{b} build supply depot at {a}"
build_barracks = lambda b, a1, a2: f"{b} build barracks at {a1} depot at {a2}"
build_factory = lambda b, a1, a2: f"{b} build factory at {a1} barracks at {a2}"
build_starport = lambda b, a1, a2: f"{b} build starport at {a1} factory at {a2}"
build_fusion_core = lambda b, a1, a2: f"{b} build fusion core at {a1} starport at {a2}"
train_marine = lambda b, a: f"{b} train marine at {a}"
train_tank = lambda b, a: f"{b} train tank at {a}"
train_wraith = lambda b, a: f"{b} train wraith at {a}"
train_battlecruiser = lambda b, a1, a2: f"{b} train battlecruiser at {a1} fusion core at {a2}"

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
    return STRIPS_domain(features, actions)

def create_StarCraft_problem(domain: STRIPS_domain, initial_state: Dict[str, bool], goal: Dict[str, bool]) -> Planning_problem:
    features = domain.feature_domain_dict.keys()
    full_initial_state = {feature: False for feature in features}
    for key, value in initial_state.items():
        full_initial_state[key] = value
    return Planning_problem(domain, full_initial_state, goal)

def get_current_level(state: Dict[str, bool]) -> int:
    if any([state[battlecruiser(area)] for area in areas]):
        return 5.5
    if any([state[fusion_core(area)] for area in areas]):
        return 5
    if any([state[wraith(area)] for area in areas]):
        return 4.5
    if any([state[starport(area)] for area in areas]):
        return 4
    if any([state[tank(area)] for area in areas]):
        return 3.5
    if any([state[factory(area)] for area in areas]):
        return 3
    if any([state[marine(area)] for area in areas]):
        return 2.5
    if any([state[barracks(area)] for area in areas]):
        return 2
    if any([state[depot(area)] for area in areas]):
        return 1
    return 0

def get_goal_level(goal: Dict[str, bool]) -> int:
    if any([battlecruiser(area) in goal for area in areas]):
        return 5.5
    if any([fusion_core(area) in goal for area in areas]):
        return 5
    if any([wraith(area) in goal for area in areas]):
        return 4.5
    if any([starport(area) in goal for area in areas]):
        return 4
    if any([tank(area) in goal for area in areas]):
        return 3.5
    if any([factory(area) in goal for area in areas]):
        return 3
    if any([marine(area) in goal for area in areas]):
        return 2.5
    if any([barracks(area) in goal for area in areas]):
        return 2
    if any([depot(area) in goal for area in areas]):
        return 1
    return 0

def heuristic(state: Dict[str, bool], goal: Dict[str, bool]) -> int:
    current_level = get_current_level(state)
    goal_level = get_goal_level(goal)

    return goal_level - current_level

if not os.path.exists("results"):
    os.makedirs("results")

problem_schemas = [
    (
        "barracks", 
        {"scv"},
        {"sectorA", "sectorB", "mineralFieldA", "mineralFieldB"},
        {
            scv("scv"): True,
            location("sectorA"): True, location("sectorB"): True,
            location("mineralFieldA"): True, location("mineralFieldB"): True,
            minerals("mineralFieldA"): True, minerals("mineralFieldB"): True,
            at("scv", "sectorA"): True
        },
        {barracks("sectorA"): True}
    ),
    (
        "barracks_with_subgoals", 
        {"scv"},
        {"sectorA", "sectorB", "mineralFieldA", "mineralFieldB", "mineralFieldC"},
        {
            scv("scv"): True,
            location("sectorA"): True, location("sectorB"): True,
            location("mineralFieldA"): True, location("mineralFieldB"): True, location("mineralFieldC"): True,
            minerals("mineralFieldA"): True, minerals("mineralFieldB"): True, minerals("mineralFieldC"): True,
            at("scv", "sectorA"): True
        },
        {depot("sectorB"): True, empty("mineralFieldC"): False, barracks("sectorA"): True}
    ),
    (
        "marine",
        {"scv"},
        {"sectorA", "sectorB", "mineralFieldA", "mineralFieldB", "mineralFieldC"},
        {
            scv("scv"): True,
            location("sectorA"): True, location("sectorB"): True,
            location("mineralFieldA"): True, location("mineralFieldB"): True, location("mineralFieldC"): True,
            minerals("mineralFieldA"): True, minerals("mineralFieldB"): True, minerals("mineralFieldC"): True,
            at("scv", "sectorA"): True
        },
        {marine("sectorA"): True}
    ),
    (
        "marine_with_subgoals",
        {"scv"},
        {"sectorA", "sectorB", "sectorC", "mineralFieldA", "mineralFieldB", "mineralFieldC", "mineralFieldD"},
        {
            scv("scv"): True,
            location("sectorA"): True, location("sectorB"): True, location("sectorC"): True,
            location("mineralFieldA"): True, location("mineralFieldB"): True, location("mineralFieldC"): True, location("mineralFieldD"): True,
            minerals("mineralFieldA"): True, minerals("mineralFieldB"): True, minerals("mineralFieldC"): True, minerals("mineralFieldD"): True,
            at("scv", "sectorA"): True
        },
        {barracks("sectorA"): True, barracks("sectorB"): True, marine("sectorA"): True}
    ),
    (
        "tank",
        {"scv"},
        {"sectorA", "sectorB", "sectorC", "mineralFieldA", "mineralFieldB", "mineralFieldC", "mineralFieldD"},
        {
            scv("scv"): True,
            location("sectorA"): True, location("sectorB"): True, location("sectorC"): True,
            location("mineralFieldA"): True, location("mineralFieldB"): True, location("mineralFieldC"): True, location("mineralFieldD"): True,
            minerals("mineralFieldA"): True, minerals("mineralFieldB"): True, minerals("mineralFieldC"): True, minerals("mineralFieldD"): True,
            at("scv", "sectorA"): True
        },
        {tank("sectorA"): True}
    ),
    (
        "tank_with_subgoals",
        {"scv"},
        {"sectorA", "sectorB", "sectorC", "mineralFieldA", "mineralFieldB", "mineralFieldC", "mineralFieldD", "mineralFieldE"},
        {
            scv("scv"): True,
            location("sectorA"): True, location("sectorB"): True, location("sectorC"): True,
            location("mineralFieldA"): True, location("mineralFieldB"): True, location("mineralFieldC"): True, location("mineralFieldD"): True, location("mineralFieldE"): True,
            minerals("mineralFieldA"): True, minerals("mineralFieldB"): True, minerals("mineralFieldC"): True, minerals("mineralFieldD"): True, minerals("mineralFieldE"): True,
            at("scv", "sectorA"): True
        },
        {barracks("sectorB"): True, marine("sectorB"): True, tank("sectorA"): True}
    )
    # ,
    # (
    #     "three_marines",
    #     {"scv"},
    #     {"sectorA", "sectorB", "sectorC", "sectorD", "mineralFieldA", "mineralFieldB", "mineralFieldC", "mineralFieldD", "mineralFieldE", "mineralFieldF", "mineralFieldG"},
    #     {
    #         scv("scv"): True,
    #         location("sectorA"): True, location("sectorB"): True, location("sectorC"): True, location("sectorD"): True,
    #         location("mineralFieldA"): True, location("mineralFieldB"): True, location("mineralFieldC"): True, location("mineralFieldD"): True, location("mineralFieldE"): True, location("mineralFieldF"): True, location("mineralFieldG"): True,
    #         minerals("mineralFieldA"): True, minerals("mineralFieldB"): True, minerals("mineralFieldC"): True, minerals("mineralFieldD"): True, minerals("mineralFieldE"): True, minerals("mineralFieldF"): True, minerals("mineralFieldG"): True,
    #         at("scv", "sectorA"): True
    #     },
    #     {marine("sectorA"): True, marine("sectorD"): True, marine("sectorC"): True}
    # )
]

for name, builders, areas, initial_state, goal in problem_schemas:
    start_time = time()
    domain = create_StarCraft_domain(builders, areas)
    problem = create_StarCraft_problem(domain, initial_state, goal)
    searcher = SearcherMPP(Forward_STRIPS(problem, heuristic))
    solution = searcher.search()
    end_time = time()

    result = f"{name}: {len(domain.feature_domain_dict)} features, {len(domain.actions)} actions; {end_time - start_time}s\n"
    for i, line in enumerate(solution.__repr__().split("\n")[1:]):
        start = line.find("--")
        end = line.find("-->")
        result += f"{i + 1}. {line[(start + 2):end]}\n"

    with open(f"results/{name}", "w") as file:
        file.write(result)

for name, builders, areas, initial_state, goal in problem_schemas:
    start_time = time()
    domain = create_StarCraft_domain(builders, areas)
    problem = create_StarCraft_problem(domain, initial_state, goal)
    searcher = SearcherMPP(Forward_STRIPS(problem))
    solution = searcher.search()
    end_time = time()

    result = f"{name} no heuristic: {len(domain.feature_domain_dict)} features, {len(domain.actions)} actions; {end_time - start_time}s\n"
    for i, line in enumerate(solution.__repr__().split("\n")[1:]):
        start = line.find("--")
        end = line.find("-->")
        result += f"{i + 1}. {line[(start + 2):end]}\n"

    with open(f"results/{name}_no_heuristic", "w") as file:
        file.write(result)