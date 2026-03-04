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
sys.path.insert(0, "/Users/s0n0497/Documents/genai/GENAI-by-IITKGP/assignments/week8")
from template import llm_function


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

    # Test Case 1-5: Time-based and Deadline Scenarios
    def test_01_albert_submitted_on_time(self):
        """Test: Albert submitted project report on time."""
        context = "Albert has been working on his project all week. He finished the final report today and submitted it to his manager before the deadline."
        question = "Did Albert submit his project report on time?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "YES", f"Expected YES for Albert submitting on time, got {result}"
        )

    def test_02_albert_submitted_late(self):
        """Test: Albert submitted project report after deadline."""
        context = "Albert has been working on his project all week. He finished the final report today and submitted it to his manager after the deadline."
        question = "Did Albert submit his project report on time?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "NO", f"Expected NO for Albert submitting late, got {result}"
        )

    def test_03_john_watered_yesterday(self):
        """Test: John watered plants yesterday morning."""
        context = "John started watering his plants every morning this week."
        question = "Did John water his plants yesterday morning?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "YES", f"Expected YES for John watering yesterday, got {result}"
        )

    def test_04_john_watered_last_month(self):
        """Test: John did not water plants last month."""
        context = "John started watering his plants every morning this week."
        question = "Did John water his plants last month?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "NO", f"Expected NO for John watering last month, got {result}"
        )

    def test_05_meeting_scheduled_today(self):
        """Test: Meeting was scheduled for today."""
        context = "The team meeting was scheduled for today at 3 PM. All members attended the meeting."
        question = "Was the meeting scheduled for today?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "YES", f"Expected YES for meeting scheduled today, got {result}"
        )

    # Test Case 6-10: Location and Geography
    def test_06_sarah_traveled_to_paris(self):
        """Test: Sarah traveled to Paris."""
        context = "Sarah booked a flight to Paris last month. She arrived in Paris on Monday and stayed for a week."
        question = "Did Sarah travel to Paris?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "YES", f"Expected YES for Sarah traveling to Paris, got {result}"
        )

    def test_07_sarah_traveled_to_london(self):
        """Test: Sarah did not travel to London."""
        context = "Sarah booked a flight to Paris last month. She arrived in Paris on Monday and stayed for a week."
        question = "Did Sarah travel to London?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "NO", f"Expected NO for Sarah traveling to London, got {result}"
        )

    def test_08_conference_in_new_york(self):
        """Test: Conference was held in New York."""
        context = "The annual technology conference was held in New York this year. Over 5000 attendees participated."
        question = "Was the conference held in New York?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "YES", f"Expected YES for conference in New York, got {result}"
        )

    def test_09_restaurant_in_downtown(self):
        """Test: Restaurant is located in downtown."""
        context = "The new Italian restaurant opened in downtown last Friday. It serves authentic pasta and pizza."
        question = "Is the restaurant located in downtown?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "YES", f"Expected YES for restaurant in downtown, got {result}"
        )

    def test_10_library_closed_on_sunday(self):
        """Test: Library is closed on Sunday."""
        context = "The public library is open Monday through Saturday from 9 AM to 6 PM. It remains closed on Sundays."
        question = "Is the library closed on Sunday?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "YES", f"Expected YES for library closed on Sunday, got {result}"
        )

    # Test Case 11-15: Actions and Activities
    def test_11_maria_completed_assignment(self):
        """Test: Maria completed her assignment."""
        context = "Maria spent the entire weekend working on her assignment. She completed it on Sunday evening and submitted it online."
        question = "Did Maria complete her assignment?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "YES", f"Expected YES for Maria completing assignment, got {result}"
        )

    def test_12_tom_attended_class(self):
        """Test: Tom attended the class."""
        context = "Tom woke up early this morning and attended his 9 AM mathematics class. He took detailed notes during the lecture."
        question = "Did Tom attend the class?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "YES", f"Expected YES for Tom attending class, got {result}"
        )

    def test_13_lisa_bought_coffee(self):
        """Test: Lisa bought coffee."""
        context = "Lisa went to the coffee shop this morning. She bought a large cappuccino and a chocolate croissant."
        question = "Did Lisa buy coffee?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "YES", f"Expected YES for Lisa buying coffee, got {result}"
        )

    def test_14_david_did_not_exercise(self):
        """Test: David did not exercise today."""
        context = "David planned to go to the gym today, but he felt unwell in the morning and decided to rest at home instead."
        question = "Did David exercise today?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "NO", f"Expected NO for David exercising, got {result}"
        )

    def test_15_emily_cooked_dinner(self):
        """Test: Emily cooked dinner."""
        context = "Emily went to the grocery store in the afternoon. She bought fresh vegetables and cooked a delicious dinner for her family."
        question = "Did Emily cook dinner?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "YES", f"Expected YES for Emily cooking dinner, got {result}"
        )

    # Test Case 16-20: Possessions and Ownership
    def test_16_james_owns_car(self):
        """Test: James owns a car."""
        context = "James bought a new red car last year. He drives it to work every day and takes good care of it."
        question = "Does James own a car?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "YES", f"Expected YES for James owning a car, got {result}"
        )

    def test_17_anna_has_laptop(self):
        """Test: Anna has a laptop."""
        context = "Anna received a new laptop as a birthday gift from her parents. She uses it for her online classes and assignments."
        question = "Does Anna have a laptop?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "YES", f"Expected YES for Anna having a laptop, got {result}"
        )

    def test_18_robert_does_not_have_bike(self):
        """Test: Robert does not have a bike."""
        context = "Robert prefers to walk to work every day. He has never owned a bicycle or motorcycle."
        question = "Does Robert have a bike?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "NO", f"Expected NO for Robert having a bike, got {result}"
        )

    def test_19_sophia_has_pet_dog(self):
        """Test: Sophia has a pet dog."""
        context = "Sophia adopted a golden retriever puppy from the animal shelter last month. She named him Max and takes him for walks every day."
        question = "Does Sophia have a pet dog?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "YES", f"Expected YES for Sophia having a pet dog, got {result}"
        )

    def test_20_michael_has_phone(self):
        """Test: Michael has a phone."""
        context = "Michael upgraded to a new smartphone last week. He uses it for calls, messages, and browsing the internet."
        question = "Does Michael have a phone?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "YES", f"Expected YES for Michael having a phone, got {result}"
        )

    # Test Case 21-25: Characteristics and States
    def test_21_cake_is_chocolate(self):
        """Test: The cake is chocolate flavored."""
        context = "The birthday cake was chocolate flavored with vanilla frosting. Everyone at the party enjoyed it."
        question = "Is the cake chocolate flavored?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "YES", f"Expected YES for chocolate cake, got {result}"
        )

    def test_22_weather_is_sunny(self):
        """Test: The weather is sunny."""
        context = "Today's weather is sunny and warm with clear blue skies. It's a perfect day for outdoor activities."
        question = "Is the weather sunny?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "YES", f"Expected YES for sunny weather, got {result}"
        )

    def test_23_book_is_fiction(self):
        """Test: The book is fiction."""
        context = "The library book I borrowed is a science fiction novel written by a famous author. It has 400 pages."
        question = "Is the book fiction?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "YES", f"Expected YES for fiction book, got {result}"
        )

    def test_24_movie_is_not_horror(self):
        """Test: The movie is not a horror film."""
        context = "We watched a romantic comedy movie last night. It was funny and heartwarming with a happy ending."
        question = "Is the movie a horror film?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "NO", f"Expected NO for horror movie, got {result}"
        )

    def test_25_room_is_clean(self):
        """Test: The room is clean."""
        context = "I spent the morning cleaning my room. I organized all the books, made the bed, and vacuumed the floor. The room is now spotless."
        question = "Is the room clean?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(
            result, "YES", f"Expected YES for clean room, got {result}"
        )


def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all tests
    suite.addTests(loader.loadTestsFromTestCase(TestLLMFunction))

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
