"""Unit test suite batch 2 for template.py."""

import sys
import unittest
import warnings

import torch
import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer

from template import llm_function

# Suppress warnings
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()


class TestLLMFunctionBatch2(unittest.TestCase):
    """Second test suite for llm_function with test cases 26-50."""

    @classmethod
    def setUpClass(cls):
        """Load model and tokenizer once for all tests."""
        print("\n" + "=" * 80)
        print("Loading Model and Tokenizer for Batch 2...")
        print("=" * 80)
        cls.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
        cls.model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-xl"
        )
        torch.manual_seed(42)
        print("Model and Tokenizer loaded successfully for Batch 2!\n")

    def setUp(self):
        torch.manual_seed(42)

    def test_26_earth_sun(self):
        questions = [
            "What is Earth?",
            "What does Earth orbit?",
            "Does Earth orbit the Sun?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "YES")

    def test_27_mars_jupiter(self):
        questions = [
            "What is Mars?",
            "Where is Mars in the solar system?",
            "Is Mars larger than Jupiter?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "NO")

    def test_28_moon_earth(self):
        questions = [
            "What is the Moon?",
            "Which planet does it orbit?",
            "Does it orbit Earth?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "YES")

    def test_29_sun_planet(self):
        questions = [
            "What is the Sun?",
            "What type of celestial body is it?",
            "Is the Sun a planet?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "NO")

    def test_30_saturn_rings(self):
        questions = [
            "What is Saturn?",
            "What is a notable visual feature of Saturn?",
            "Does Saturn have rings?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "YES")

    def test_31_two_plus_two(self):
        questions = [
            "What is 2 + 2?",
            "Is that result an even number?",
            "Is 2 + 2 equal to 4?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "YES")

    def test_32_prime_four(self):
        questions = [
            "What is a prime number?",
            "How many divisors does 4 have?",
            "Is 4 a prime number?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "NO")

    def test_33_triangle_sides(self):
        questions = [
            "What is a triangle?",
            "How many sides does it have?",
            "Does a triangle have three sides?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "YES")

    def test_34_square_three_sides(self):
        questions = [
            "What is a square?",
            "How many sides does it have?",
            "Does a square have three sides?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "NO")

    def test_35_ten_greater_than_five(self):
        questions = [
            "What are the numbers 10 and 5?",
            "Which one is larger?",
            "Is 10 greater than 5?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "YES")

    def test_36_humans_mammals(self):
        questions = [
            "What are humans in biological classification?",
            "What class do they belong to?",
            "Are humans mammals?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "YES")

    def test_37_spiders_insects(self):
        questions = [
            "What is a spider?",
            "How many legs does it usually have?",
            "Is a spider an insect?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "NO")

    def test_38_heart_pumps_blood(self):
        questions = [
            "What is the heart?",
            "What is its primary function?",
            "Does the heart pump blood?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "YES")

    def test_39_plants_photosynthesis(self):
        questions = [
            "What do green plants do with sunlight?",
            "What is that process called?",
            "Do plants perform photosynthesis?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "YES")

    def test_40_whale_fish(self):
        questions = [
            "What is a whale?",
            "How does it breathe?",
            "Is a whale a fish?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "NO")

    def test_41_cpu_computer(self):
        questions = [
            "What is a CPU?",
            "What role does it play in a computer?",
            "Is CPU short for Central Processing Unit?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "YES")

    def test_42_html_programming_language(self):
        questions = [
            "What is HTML?",
            "What is it used for?",
            "Is HTML a programming language?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "NO")

    def test_43_email_internet(self):
        questions = [
            "What is email?",
            "How is it commonly transmitted?",
            "Does email use the internet?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "YES")

    def test_44_ram_storage(self):
        questions = [
            "What is RAM in a computer?",
            "Is it typically volatile or permanent?",
            "Is RAM permanent long-term storage?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "NO")

    def test_45_python_interpreted(self):
        questions = [
            "What is Python in computing?",
            "How is Python code usually executed?",
            "Is Python generally an interpreted language?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "YES")

    def test_46_ice_hotter_than_fire(self):
        questions = [
            "What is ice?",
            "What is fire?",
            "Is ice hotter than fire?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "NO")

    def test_47_day_after_monday(self):
        questions = [
            "What comes after Monday?",
            "What day is that in a weekly sequence?",
            "Is Tuesday the day after Monday?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "YES")

    def test_48_penguin_can_fly(self):
        questions = [
            "What is a penguin?",
            "How does it usually move in water and on land?",
            "Can penguins fly in the sky like eagles?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "NO")

    def test_49_boiling_water_100c(self):
        questions = [
            "What is boiling point?",
            "At standard pressure, what is water's boiling point in Celsius?",
            "Does water boil at 100 degrees Celsius at sea level?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "YES")

    def test_50_honey_bees(self):
        questions = [
            "What is honey?",
            "Which creatures commonly produce it?",
            "Is honey produced by bees?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "YES")


def run_tests_batch_2():
    """Run all batch 2 tests with detailed output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestLLMFunctionBatch2))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("BATCH 2 TEST SUMMARY")
    print("=" * 80)
    print(f"Tests Run: {result.testsRun}")
    successes = result.testsRun - len(result.failures) - len(result.errors)
    print(f"Successes: {successes}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80 + "\n")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests_batch_2()
    sys.exit(0 if success else 1)
