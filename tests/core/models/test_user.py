import pytest
from core.models import User
from django.core.exceptions import ValidationError


@pytest.fixture
def empty_user():
    return User()


class TestInvalidUsers():
    @pytest.mark.django_db
    def test_empty_user(self, empty_user):
        with pytest.raises(ValidationError):
            empty_user.full_clean()

    @pytest.mark.django_db
    def test_user_without_password(self):
        with pytest.raises(ValidationError):
            User(username="test_username").full_clean()

    @pytest.mark.django_db
    def test_user_without_username(self):
        with pytest.raises(ValidationError):
            User(password="test password").full_clean()

    @pytest.mark.django_db
    def test_username_with_invalid_characters(self):
        with pytest.raises(ValidationError):
            User(password="password", username="$test").full_clean()

    @pytest.mark.django_db
    def test_username_too_long(self):
        with pytest.raises(ValidationError):
            User(password="password", username=("X"*151)).full_clean()
