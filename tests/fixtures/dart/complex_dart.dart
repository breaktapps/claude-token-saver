import 'dart:async';
import 'package:flutter/material.dart';

/// SubscriptionProvider com generics complexos e annotations
class SubscriptionProvider extends ChangeNotifier with LoggingMixin {
  final String _id;
  final Repository _repository;

  SubscriptionProvider(this._id, this._repository);

  SubscriptionProvider.fromId(String id, Repository repo)
      : _id = id,
        _repository = repo;

  factory SubscriptionProvider.create(Repository repo) {
    return SubscriptionProvider('default', repo);
  }

  String get subscriptionId => _id;

  set subscriptionId(String value) {
    _id = value;
    notifyListeners();
  }

  @override
  Future<Either<Failure, List<Feature>>> loadFeatures() async {
    final result = await _repository.fetch(_id);
    notifyListeners();
    return result;
  }

  @visibleForTesting
  void resetState() {
    notifyListeners();
  }
}

/// Widget com generics em build
class FeatureGate extends StatelessWidget {
  final String featureId;

  const FeatureGate({super.key, required this.featureId});

  @override
  Widget build(BuildContext context) {
    return Consumer<SubscriptionProvider>(
      builder: (context, provider, child) {
        return Text(provider.subscriptionId);
      },
    );
  }

  Widget overlay(BuildContext context) {
    return Container(
      child: Text(featureId),
    );
  }
}

/// Extension method
extension StringExt on String {
  String capitalize() {
    if (isEmpty) return this;
    return this[0].toUpperCase() + substring(1);
  }

  bool get isEmail => contains('@');
}

/// Mixin
mixin LoggingMixin {
  void log(String msg) {
    print('[LOG] $msg');
  }

  void logError(String msg, Object error) {
    print('[ERROR] $msg: $error');
  }
}

/// Top-level function
Future<void> initializeApp(String config) async {
  final provider = SubscriptionProvider.create(Repository());
  await provider.loadFeatures();
}
