import unittest

import numpy as np

from road_tour.anneal import simulated_annealing
from road_tour.network import CITIES, EDGE_LENGTH, has_road
from road_tour.qubo import build_qubo, decode, direct_energy, encode, exhaustive, is_legal
from road_tour.tours import all_visit_orders, constrained_tours, on_road_tours, shortest, uses_road


class TourTests(unittest.TestCase):
    def test_twenty_four_visit_orders(self):
        self.assertEqual(len(all_visit_orders()), 24)

    def test_known_feasible_lengths(self):
        constrained = constrained_tours()
        lengths = sorted(tour.length for tour in constrained)
        self.assertEqual(lengths, [16, 19])
        orders = {tour.order for tour in constrained}
        self.assertEqual(orders, {(0, 4, 2, 3, 1), (0, 4, 2, 1, 3)})

    def test_optimum_is_the_shorter_feasible_tour(self):
        best = shortest(constrained_tours())
        self.assertEqual(best.order, (0, 4, 2, 3, 1))
        self.assertEqual(best.length, 16)

    def test_depot_to_city_two_is_not_used_by_a_constrained_tour(self):
        for tour in constrained_tours():
            self.assertFalse(uses_road(tour, 0, 2))

    def test_city_two_immediately_after_the_depot_has_no_on_road_tour(self):
        placed_first = [
            tour for tour in on_road_tours()
            if tour.order[1] == 2
        ]
        self.assertEqual(placed_first, [])

    def test_city_four_lies_between_the_depot_and_city_two(self):
        for tour in on_road_tours():
            position = tour.order.index(4)
            left = tour.order[(position - 1) % len(CITIES)]
            right = tour.order[(position + 1) % len(CITIES)]
            self.assertEqual({left, right}, {0, 2})

    def test_constrained_tours_leave_through_city_four(self):
        for tour in constrained_tours():
            self.assertEqual(tour.order[:3], (0, 4, 2))

    def test_roads_are_symmetric_and_loopless(self):
        for pair in EDGE_LENGTH:
            a, b = tuple(pair)
            self.assertNotEqual(a, b)
            self.assertTrue(has_road(a, b))
            self.assertTrue(has_road(b, a))
            self.assertFalse(has_road(a, a))


class QuboTests(unittest.TestCase):
    def test_quadratic_form_matches_the_hamiltonian_on_every_state(self):
        qubo = build_qubo(onehot_penalty=25, road_penalty=25)
        n = len(qubo.variables)
        for mask in range(1 << n):
            state = np.array([(mask >> bit) & 1 for bit in range(n)], dtype=int)
            left = qubo.energy(state)
            right = direct_energy(state, 25, 25)
            self.assertAlmostEqual(left, right, places=8)

    def test_legal_optimum_has_energy_equal_to_its_length(self):
        qubo = build_qubo(onehot_penalty=25, road_penalty=25)
        ground, _ = exhaustive(qubo)
        self.assertTrue(ground.legal)
        self.assertEqual(ground.order, (0, 4, 2, 3, 1))
        self.assertAlmostEqual(ground.energy, 16.0)

    def test_zero_penalty_does_not_select_the_tour(self):
        qubo = build_qubo(onehot_penalty=0, road_penalty=0)
        ground, _ = exhaustive(qubo)
        self.assertFalse(ground.legal)

    def test_encode_round_trip(self):
        for order in ((0, 4, 2, 3, 1), (0, 4, 2, 1, 3)):
            state = encode(order)
            self.assertEqual(decode(state), order)
            self.assertTrue(is_legal(state))

    def test_penalty_crosses_seven(self):
        qubo = build_qubo(7, 7)
        optimum = encode((0, 4, 2, 3, 1))
        ghost = encode((0, 1, 2, 3, 4))
        self.assertAlmostEqual(qubo.energy(optimum), 16.0)
        self.assertAlmostEqual(qubo.energy(ghost), 16.0)

        qubo = build_qubo(8, 8)
        ground, energies = exhaustive(qubo)
        self.assertEqual(ground.order, (0, 4, 2, 3, 1))
        self.assertEqual(int(np.sum(np.abs(energies - ground.energy) < 1e-8)), 1)

    def test_a_bit_flip_off_the_optimum_costs_at_least_42(self):
        qubo = build_qubo(25, 25)
        state = encode((0, 4, 2, 3, 1))
        base = qubo.energy(state)
        deltas = []
        for bit in range(len(state)):
            state[bit] = 1 - state[bit]
            deltas.append(qubo.energy(state) - base)
            state[bit] = 1 - state[bit]
        self.assertGreaterEqual(min(deltas), 42.0)

    def test_annealing_reaches_the_ground_state(self):
        qubo = build_qubo(25, 25)
        run = simulated_annealing(qubo, steps=2000, t_start=40.0, t_end=0.05, seed=20)
        self.assertAlmostEqual(run.best_energy, 16.0)
        self.assertEqual(decode(run.best_state), (0, 4, 2, 3, 1))

    def test_decode_rejects_an_empty_state(self):
        state = np.zeros(9, dtype=int)
        self.assertIsNone(decode(state))
        self.assertFalse(is_legal(state))


if __name__ == "__main__":
    unittest.main()
