"""Unit test suite batch 2 for template.py."""

import sys
import unittest
import warnings

import torch
import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Import the function to test
sys.path.insert(0, "/Users/s0n0497/Documents/genai/GENAI-by-IITKGP/assignments/week8")
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

    def test_26_student_passed_exam(self):
        """Test: Student passed the exam."""
        context = "The student studied hard for the final exam. She scored 85 out of 100, which is above the passing grade of 60."
        question = "Did the student pass the exam?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_27_team_won_match(self):
        """Test: Team won the match."""
        context = "The home team played exceptionally well in yesterday's match. They scored 3 goals while the opponent scored only 1 goal."
        question = "Did the home team win the match?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_28_store_is_open(self):
        """Test: Store is open on weekdays."""
        context = "The grocery store operates Monday through Friday from 8 AM to 8 PM. Today is Wednesday."
        question = "Is the store open today?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_29_package_arrived(self):
        """Test: Package arrived on time."""
        context = "I ordered a package online last week. The delivery was scheduled for Monday, and it arrived on Monday morning as expected."
        question = "Did the package arrive on time?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_30_meeting_was_cancelled(self):
        """Test: Meeting was cancelled."""
        context = "The board meeting scheduled for this afternoon was cancelled due to bad weather. All participants were notified via email."
        question = "Was the meeting cancelled?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_31_restaurant_serves_vegetarian(self):
        """Test: Restaurant serves vegetarian food."""
        context = "The new restaurant offers a wide variety of dishes including vegetarian, vegan, and meat options. They have a dedicated vegetarian menu."
        question = "Does the restaurant serve vegetarian food?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_32_ticket_was_expensive(self):
        """Test: Ticket was expensive."""
        context = "The concert ticket cost 200 dollars, which is much higher than the usual price of 50 dollars for similar events."
        question = "Was the ticket expensive?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_33_car_needs_repair(self):
        """Test: Car needs repair."""
        context = "My car broke down yesterday on the highway. The mechanic said the engine needs major repairs that will take several days."
        question = "Does the car need repair?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_34_doctor_recommended_rest(self):
        """Test: Doctor recommended rest."""
        context = "After the medical examination, the doctor advised the patient to take complete rest for a week and avoid any strenuous activities."
        question = "Did the doctor recommend rest?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_35_project_was_completed(self):
        """Test: Project was completed on schedule."""
        context = "The software development project was completed on schedule last Friday. All features were implemented and tested successfully."
        question = "Was the project completed on schedule?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_36_friend_visited_last_week(self):
        """Test: Friend visited last week."""
        context = "My best friend came to visit me last week. We spent three days together exploring the city and catching up."
        question = "Did my friend visit last week?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_37_recipe_requires_eggs(self):
        """Test: Recipe requires eggs."""
        context = "The cake recipe calls for three eggs, two cups of flour, one cup of sugar, and butter. All ingredients must be fresh."
        question = "Does the recipe require eggs?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_38_hotel_has_pool(self):
        """Test: Hotel has a swimming pool."""
        context = "The luxury hotel features a large outdoor swimming pool, a fitness center, and a spa. Guests can use all facilities free of charge."
        question = "Does the hotel have a swimming pool?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_39_train_was_delayed(self):
        """Test: Train was delayed."""
        context = "The morning train was supposed to arrive at 9 AM, but due to technical issues, it arrived at 10:30 AM instead."
        question = "Was the train delayed?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_40_book_was_returned(self):
        """Test: Book was returned to library."""
        context = "I borrowed a novel from the library two weeks ago. I finished reading it and returned it to the library yesterday before the due date."
        question = "Was the book returned to the library?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_41_concert_was_outdoors(self):
        """Test: Concert was held outdoors."""
        context = "The summer music festival was held in the city park, which is an outdoor venue. Thousands of people attended the event."
        question = "Was the concert held outdoors?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_42_course_has_prerequisite(self):
        """Test: Course has a prerequisite."""
        context = "The advanced mathematics course requires students to complete Calculus 101 as a prerequisite. Without it, enrollment is not allowed."
        question = "Does the course have a prerequisite?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_43_party_was_surprise(self):
        """Test: Party was a surprise."""
        context = "My friends organized a surprise birthday party for me. They kept it secret for weeks and I had no idea about it until I arrived."
        question = "Was the party a surprise?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_44_job_requires_experience(self):
        """Test: Job requires experience."""
        context = "The job posting for senior software engineer requires at least five years of professional experience in software development."
        question = "Does the job require experience?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_45_museum_allows_photography(self):
        """Test: Museum allows photography."""
        context = "The art museum has a policy that allows visitors to take photographs in most galleries, except for special exhibitions where photography is prohibited."
        question = "Does the museum allow photography?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_46_recipe_is_vegetarian(self):
        """Test: Recipe is vegetarian."""
        context = "The pasta recipe uses only vegetables, olive oil, and herbs. It contains no meat, fish, or animal products of any kind."
        question = "Is the recipe vegetarian?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_47_event_was_free(self):
        """Test: Event was free to attend."""
        context = "The community workshop on gardening was completely free to attend. No registration fee or ticket purchase was required."
        question = "Was the event free to attend?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_48_park_has_playground(self):
        """Test: Park has a playground."""
        context = "The neighborhood park features a large playground with swings, slides, and climbing equipment. It's popular among local children."
        question = "Does the park have a playground?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_49_class_was_cancelled(self):
        """Test: Class was cancelled."""
        context = "The professor sent an email this morning informing all students that today's afternoon class has been cancelled due to illness."
        question = "Was the class cancelled?"
        result = llm_function(self.model, self.tokenizer, context, question)
        self.assertEqual(result, "YES")

    def test_50_gift_was_wrapped(self):
        """Test: Gift was wrapped."""
        context = "I bought a birthday gift for my sister and carefully wrapped it in colorful paper with a ribbon. I will give it to her tomorrow."
        question = "Was the gift wrapped?"
        result = llm_function(self.model, self.tokenizer, context, question)
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
