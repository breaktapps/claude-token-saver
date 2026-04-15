import 'dart:convert';
import 'package:flutter/material.dart';

String formatDate(DateTime date) {
  final parts = date.toIso8601String().split('T');
  return parts[0];
}

int clamp(int value, int min, int max) {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

class DateHelper {
  String format(DateTime date) {
    return formatDate(date);
  }
}
