class AudioService {
  final AudioPlayer _player;
  final AudioCache _cache;

  AudioService(this._player, this._cache);

  void initialize() {
    _player.setVolume(0.8);
    _cache.loadDefaults();
    emit('initialized');
  }

  Future<void> processAudio(AudioData data) async {
    final decoded = decodeAudio(data);
    final filtered = applyFilter(decoded);
    await _player.play(filtered);
    emit('processed');
  }

  void dispose() {
    _player.stop();
    _cache.clear();
    emit('disposed');
  }
}
