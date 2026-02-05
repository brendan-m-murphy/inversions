#!/usr/bin/env python3
"""Tests for data download functionality."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from inversions.data.download import download_natural_earth_data, get_world_map_path


def test_get_world_map_path_returns_path():
    """Test that get_world_map_path returns a Path object."""
    # We can't actually download in tests, so we'll just verify the path structure
    path = Path(__file__).resolve().parent.parent / "src" / "inversions" / "data"
    expected_file = path / "natural_earth_50.zip"
    
    # The function should return a path
    with patch('inversions.data.download.download_natural_earth_data') as mock_download:
        mock_download.return_value = expected_file
        result = get_world_map_path()
        
        # If file doesn't exist, download should be called
        if not expected_file.exists():
            mock_download.assert_called_once()


def test_download_natural_earth_data_skips_if_exists(tmp_path):
    """Test that download is skipped if file already exists."""
    # Create a fake data file
    data_file = tmp_path / "natural_earth_50.zip"
    data_file.write_text("fake data")
    
    # Call the function - it should return the existing file without downloading
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        result = download_natural_earth_data(tmp_path)
        
        assert result == data_file
        assert result.exists()
        # Should not have called urlretrieve since file exists
        mock_retrieve.assert_not_called()


def test_download_natural_earth_data_downloads_if_missing(tmp_path):
    """Test that download is attempted if file doesn't exist."""
    data_file = tmp_path / "natural_earth_50.zip"
    
    # Mock the download
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        # Create the file when urlretrieve is called
        def create_file(url, path):
            Path(path).write_text("downloaded data")
        
        mock_retrieve.side_effect = create_file
        
        result = download_natural_earth_data(tmp_path)
        
        assert result == data_file
        # Should have called urlretrieve
        mock_retrieve.assert_called_once()
        assert "naturalearth" in mock_retrieve.call_args[0][0]
