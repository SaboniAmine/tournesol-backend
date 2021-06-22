from django.core.management.base import BaseCommand, CommandError

from tournesol.models import Comparison
from settings.settings import VIDEO_FIELDS


def fetch_data():
    """
    Fetches the data from the Comparisons model

    Returns:
    - comparison_data: list of [contributor_id: int, video_id_1: int, video_id_2: int, criteria: str, score: float, weight: float]
    """
    comparison_data = [
        [comparison.user_id, comparison.video_1_id, comparison.video_2_id, criteria, getattr(comparison, criteria), getattr(comparison, f"{criteria}_weight")]
        for comparison in Comparison.objects.all() for criteria in VIDEO_FIELDS
        if hasattr(comparison, criteria)]
    
    return comparison_data


def ml_run(comparison_data):
    """
    Uses data loaded

    Returns:
    - video_scores: list of [video_id: int, criteria_name: str, score: float, uncertainty: float]
    - contributor_rating_scores: list of [contributor_id: int, video_id: int, criteria_name: str, score: float, uncertainty: float]
    """
    # TODO: implement this method
    video_scores = [[1,"reliability", 1., 1.]]
    contributor_rating_scores = [[1, 1, "reliability", 1., 1.]]
    return video_scores, contributor_rating_scores


def save_data(video_scores, contributor_rating_scores):
    """
    Saves in the scores for Videos and ContributorRatings
    """
    pass


class Command(BaseCommand):
    help = 'Runs the '

    def handle(self, *args, **options):
        comparison_data = fetch_data()
        video_scores, contributor_rating_scores = ml_run(comparison_data)
        save_data(video_scores, contributor_rating_scores)
