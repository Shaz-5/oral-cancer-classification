import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_tflite/flutter_tflite.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:developer' as devtools;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Oral Cancer Detection',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? filePath;
  String label = '';
  double confidence = 0.0;
  bool isLoading = false; // Track loading state

  Future<void> _tfLiteInit() async {
    try {
      await Tflite.loadModel(
        model: "assets/model.tflite",
        labels: "assets/labels.txt",
      );
    } catch (e) {
      devtools.log("Error loading model: $e");
      _showErrorDialog("Failed to load model.");
    }
  }

  Future<void> _selectImage(ImageSource source) async {
    final picker = ImagePicker();
    try {
      final XFile? image = await picker.pickImage(source: source);
      if (image == null) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('No image selected')),
        );
        return;
      }

      setState(() {
        filePath = File(image.path);
        isLoading = true; // Set loading state
      });

      devtools.log("Running model on image: ${image.path}");

      var recognitions = await Tflite.runModelOnImage(
        path: image.path,
        imageMean: 0.0,
        imageStd: 255.0,
        numResults: 3,
        threshold: 0.2,
        asynch: true,
      );

      if (recognitions != null && recognitions.isNotEmpty) {
        setState(() {
          confidence = recognitions[0]['confidence'] * 100;
          label = recognitions[0]['label'].toString();
        });
      } else {
        _showErrorDialog("No predictions made.");
      }
    } catch (e) {
      _showErrorDialog("Error selecting image: $e");
    } finally {
      setState(() {
        isLoading = false; // Reset loading state
      });
    }
  }

  void _clearImage() {
    setState(() {
      filePath = null;
      label = '';
      confidence = 0.0;
    });
  }

  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Error'),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  Color _getLabelColor(String label) {
    switch (label) {
      case 'Normal':
        return Colors.green; // Color for normal
      case 'Oral Cancer':
        return Colors.redAccent; // Color for oral cancer
      case 'Pre Cancer':
        return Colors.deepOrangeAccent; // Dark orange for pre cancer
      default:
        return Colors.black; // Default color
    }
  }

  @override
  void dispose() {
    super.dispose();
    Tflite.close();
  }

  @override
  void initState() {
    super.initState();
    _tfLiteInit();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Oral Cancer Detection"),
      ),
      body: SingleChildScrollView(
        child: Center(
          child: Column(
            children: [
              const SizedBox(height: 36),
              Card(
                elevation: 10,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(15),
                ),
                child: Container(
                  width: 300,
                  padding: const EdgeInsets.all(24), // Increased padding
                  child: Column(
                    children: [
                      Stack(
                        alignment: Alignment.topRight,
                        children: [
                          Container(
                            height: 250,
                            width: 250,
                            decoration: BoxDecoration(
                              borderRadius: BorderRadius.circular(12),
                              color: Colors.grey[200],
                            ),
                            child: filePath == null
                                ? const Center(child: Text('No Image Selected'))
                                : ClipRRect(
                                    borderRadius: BorderRadius.circular(12),
                                    child: Image.file(
                                      filePath!,
                                      fit: BoxFit.cover,
                                    ),
                                  ),
                          ),
                          if (filePath != null)
                            Positioned(
                              right: 8,
                              top: 8,
                              child: GestureDetector(
                                onTap: _clearImage,
                                child: Container(
                                  decoration: BoxDecoration(
                                    color: Colors.grey.withOpacity(0.7),
                                    shape: BoxShape.circle,
                                  ),
                                  padding: const EdgeInsets.all(6),
                                  child: const Icon(
                                    Icons.clear,
                                    color: Colors.white,
                                    size: 24,
                                  ),
                                ),
                              ),
                            ),
                        ],
                      ),
                      const SizedBox(height: 30), // Increased spacing
                      Text(
                        label,
                        style: TextStyle(
                          fontSize: 24, // Increased font size
                          fontWeight: FontWeight.bold,
                          color: _getLabelColor(label), // Get color based on label
                        ),
                      ),
                      const SizedBox(height: 28), // Increased spacing
                      Text(
                        "Confidence: ${confidence.toStringAsFixed(0)}%",
                        style: const TextStyle(fontSize: 18), // Increased font size
                      ),
                      const SizedBox(height: 18), // Increased spacing
                      if (isLoading) const CircularProgressIndicator(),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 56), // Increased spacing
              ElevatedButton(
                onPressed: () => _selectImage(ImageSource.camera),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 16), // Increased padding
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: const Text("Take a Photo"),
              ),
              const SizedBox(height: 36), // Increased spacing
              ElevatedButton(
                onPressed: () => _selectImage(ImageSource.gallery),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 16), // Increased padding
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: const Text("Select from Gallery"),
              ),
              const SizedBox(height: 30), // Extra bottom space
            ],
          ),
        ),
      ),
    );
  }
}
