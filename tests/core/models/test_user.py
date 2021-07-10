import pytest
from core.models import User
from django.core.exceptions import ValidationError


@pytest.fixture
def empty_user():
    return User()


@pytest.fixture
def user():
    def inner(**kwargs):
        return User(**kwargs)
    return inner


class TestInvalidUsers():
    @pytest.mark.django_db
    def test_empty_user(self, empty_user):
        with pytest.raises(ValidationError):
            empty_user.full_clean()

    @pytest.mark.django_db
    def test_user_without_password(self, user):
        with pytest.raises(ValidationError):
            user(username="test_username").full_clean()

    @pytest.mark.django_db
    def test_user_without_username(self, user):
        with pytest.raises(ValidationError):
            user(password="test password").full_clean()

    @pytest.mark.django_db
    def test_username_with_invalid_characters(self, user):
        with pytest.raises(ValidationError):
            user(password="password", username="$test").full_clean()

    @pytest.mark.django_db
    def test_username_too_long(self, user):
        with pytest.raises(ValidationError):
            user(password="password", username=("X"*151)).full_clean()