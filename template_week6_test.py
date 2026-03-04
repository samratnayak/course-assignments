"""
Unit test suite for template.py
Tests the llm_function with actual model output using unittest framework.
25 comprehensive test cases covering various topics and scenarios.
"""

import unittest
import sys
import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

# Import the function to test
#sys.path.insert(0, "/Users/s0n0497/Documents/genai/GENAI-by-IITKGP/assignments/week6")
from template import llm_function
#, get_model_response


class TestLLMFunction(unittest.TestCase):
    """Test suite for llm_function with 25 test cases."""

    @classmethod
    def setUpClass(cls):
        """Load model and tokenizer once for all tests."""
        print("\n" + "=" * 80)
        print("Loading Model and Tokenizer...")
        print("=" * 80)
        cls.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
        cls.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
        torch.manual_seed(42)
        print("Model and Tokenizer loaded successfully!\n")

    def setUp(self):
        """Set random seed before each test."""
        torch.manual_seed(42)

    # Test Case 1-5: Historical Figures and Geography
    def test_01_tagore_america(self):
        """Test: Rabindranath Tagore - Is he from America?"""
        questions = [
            "Who is Rabindranath Tagore?",
            "Where was he born?",
            "Is it in America?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result, "NO", f"Expected NO for Tagore in America, got {result}"
        )

    def test_02_tagore_india(self):
        """Test: Rabindranath Tagore - Is he from India?"""
        questions = [
            "Who is Rabindranath Tagore?",
            "Where was he born?",
            "Is it in India?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result, "YES", f"Expected YES for Tagore in India, got {result}"
        )

    def test_03_einstein_germany(self):
        """Test: Albert Einstein - Is he from Germany?"""
        questions = [
            "Who is Albert Einstein?",
            "Where was he born?",
            "Is it in Germany?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result, "YES", f"Expected YES for Einstein in Germany, got {result}"
        )

    def test_04_einstein_france(self):
        """Test: Albert Einstein - Is he from France?"""
        questions = [
            "Who is Albert Einstein?",
            "Where was he born?",
            "Is it in France?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result, "NO", f"Expected NO for Einstein in France, got {result}"
        )

    def test_05_newton_england(self):
        """Test: Isaac Newton - Is he from England?"""
        questions = ["Who is Isaac Newton?", "Where was he born?", "Is it in England?"]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result, "YES", f"Expected YES for Newton in England, got {result}"
        )

    # Test Case 6-10: Science and Nature
    def test_06_water_oxygen(self):
        """Test: Water - Does it contain oxygen?"""
        questions = [
            "What is water made of?",
            "What are its main elements?",
            "Is oxygen one of them?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result, "YES", f"Expected YES for water containing oxygen, got {result}"
        )

    def test_07_water_nitrogen(self):
        """Test: Water - Does it contain nitrogen?"""
        questions = [
            "What is water made of?",
            "What are its main chemical elements?",
            "Is nitrogen one of them?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result, "NO", f"Expected NO for water containing nitrogen, got {result}"
        )

    def test_08_oxygen_breathing(self):
        """Test: Oxygen - Is it needed for breathing?"""
        questions = [
            "What is oxygen?",
            "What is its role in nature?",
            "Do humans need it for breathing?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result, "YES", f"Expected YES for oxygen in breathing, got {result}"
        )

    def test_09_carbon_element(self):
        """Test: Carbon - Is it an element?"""
        questions = [
            "What is carbon?",
            "What type of substance is it?",
            "Is it a chemical element?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result, "YES", f"Expected YES for carbon being an element, got {result}"
        )

    def test_10_diamond_carbon(self):
        """Test: Diamond - Is it made of carbon?"""
        questions = [
            "What is a diamond?",
            "What is it composed of?",
            "Is it made of carbon?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result, "YES", f"Expected YES for diamond made of carbon, got {result}"
        )

    # Test Case 11-15: Geography and Capitals
    def test_11_france_paris(self):
        """Test: Paris - Is it the capital of France?"""
        questions = [
            "What is the capital of France?",
            "What continent is it in?",
            "Is it in Europe?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result, "YES", f"Expected YES for Paris in Europe, got {result}"
        )

    def test_12_tokyo_japan(self):
        """Test: Tokyo - Is it in Japan?"""
        questions = [
            "What is the capital of Japan?",
            "What continent is it in?",
            "Is it in Asia?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(result, "YES", f"Expected YES for Tokyo in Asia, got {result}")

    def test_13_canberra_australia(self):
        """Test: Canberra - Is it in Australia?"""
        questions = [
            "What is the capital of Australia?",
            "Is it a city in the Southern Hemisphere?",
            "Is it in Australia?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result, "YES", f"Expected YES for Canberra in Australia, got {result}"
        )

    def test_14_pacific_ocean(self):
        """Test: Pacific Ocean - Is it the largest ocean?"""
        questions = [
            "What is the largest ocean on Earth?",
            "Which continents border it?",
            "Does it border Asia?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result, "YES", f"Expected YES for Pacific bordering Asia, got {result}"
        )

    def test_15_sahara_africa(self):
        """Test: Sahara Desert - Is it in Africa?"""
        questions = [
            "What is the largest hot desert in the world?",
            "Where is it located?",
            "Is it in Africa?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result, "YES", f"Expected YES for Sahara in Africa, got {result}"
        )

    # Test Case 16-20: Technology and Inventions
    def test_16_telephone_invention(self):
        """Test: Telephone - Invented in 19th century?"""
        questions = [
            "Who invented the telephone?",
            "When did this happen?",
            "Was it in the 19th century?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result, "YES", f"Expected YES for telephone in 19th century, got {result}"
        )

    def test_17_python_programming(self):
        """Test: Python - Is it a programming language?"""
        questions = [
            "What is Python?",
            "What field is it used in?",
            "Is it a programming language?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result,
            "YES",
            f"Expected YES for Python being a programming language, got {result}",
        )

    def test_18_internet_modern(self):
        """Test: Internet - Is it a modern technology?"""
        questions = [
            "What is the Internet?",
            "When was it developed?",
            "Is it a 20th century invention?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result, "YES", f"Expected YES for Internet in 20th century, got {result}"
        )

    def test_19_lightbulb_edison(self):
        """Test: Lightbulb - Did Edison invent it?"""
        questions = [
            "What is an incandescent lightbulb?",
            "Who developed the practical version?",
            "Was it Thomas Edison?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result, "YES", f"Expected YES for Edison and lightbulb, got {result}"
        )

    def test_20_steam_engine_watt(self):
        """Test: Steam Engine - Did James Watt improve it?"""
        questions = [
            "What is a steam engine?",
            "Who made significant improvements to it?",
            "Was it James Watt?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result, "YES", f"Expected YES for Watt and steam engine, got {result}"
        )

    # Test Case 21-25: Literature, Arts, and Sports
    def test_21_shakespeare_english(self):
        """Test: Shakespeare - Is he English?"""
        questions = [
            "Who wrote Romeo and Juliet?",
            "What was his nationality?",
            "Was he English?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result, "YES", f"Expected YES for Shakespeare being English, got {result}"
        )

    def test_22_mona_lisa_da_vinci(self):
        """Test: Mona Lisa - Did Leonardo da Vinci paint it?"""
        questions = [
            "What is the Mona Lisa painting?",
            "Who painted it?",
            "Was it Leonardo da Vinci?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result, "YES", f"Expected YES for da Vinci and Mona Lisa, got {result}"
        )

    def test_23_cricket_popular_india(self):
        """Test: Cricket - Is it popular in India?"""
        questions = [
            "What is cricket?",
            "What countries play it extensively?",
            "Is it popular in India?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result,
            "YES",
            f"Expected YES for cricket being popular in India, got {result}",
        )

    def test_24_olympic_games_ancient(self):
        """Test: Olympic Games - Originated in ancient times?"""
        questions = [
            "What are the Olympic Games?",
            "Where did they originate?",
            "Did they originate in ancient Greece?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result, "YES", f"Expected YES for Olympics in ancient Greece, got {result}"
        )

    def test_25_fifa_world_cup_football(self):
        """Test: FIFA World Cup - Is it for football?"""
        questions = [
            "What is the FIFA World Cup?",
            "What sport is it for?",
            "Is it for football/soccer?",
        ]
        result = llm_function(self.model, self.tokenizer, questions)
        self.assertEqual(
            result,
            "YES",
            f"Expected YES for FIFA World Cup being for football, got {result}",
        )


# class TestGetModelResponse(unittest.TestCase):
#     """Test suite for get_model_response function."""

#     @classmethod
#     def setUpClass(cls):
#         """Load model and tokenizer once for all tests."""
#         cls.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
#         cls.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
#         torch.manual_seed(42)

#     def test_simple_question(self):
#         """Test basic response generation."""
#         question = "What is 2+2?"
#         response = get_model_response(question, self.model, self.tokenizer)
#         self.assertIsInstance(response, str)
#         self.assertGreater(len(response), 0)

#     def test_context_awareness(self):
#         """Test that model can handle context."""
#         context_question = "The sky is blue. What color is the sky?"
#         response = get_model_response(context_question, self.model, self.tokenizer)
#         self.assertIsInstance(response, str)
#         self.assertGreater(len(response), 0)


def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all tests
    suite.addTests(loader.loadTestsFromTestCase(TestLLMFunction))
    #suite.addTests(loader.loadTestsFromTestCase(TestGetModelResponse))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80 + "\n")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
