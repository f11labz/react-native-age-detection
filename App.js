/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 *
 * @format
 * @flow
 */

import React, { Component } from 'react';
import {
  SafeAreaView,
  StyleSheet,
  ScrollView,
  View,
  Text,
  StatusBar,
  NativeModules,
  Dimensions,
  Button,
} from 'react-native';

import {
  Header,
  LearnMoreLinks,
  Colors,
  DebugInstructions,
  ReloadInstructions,
} from 'react-native/Libraries/NewAppScreen';
import { RNCamera } from 'react-native-camera';

let AgeModule = NativeModules.VilaAgeModule;

const window = Dimensions.get('window');
const viewPort = {
  width: window.width <= window.height ? window.width - 10 : window.height - 10,
  height: window.width <= window.height ? window.width - 10 : window.height - 10
}

class App extends Component {

  constructor() {
    super();
    this.state = {
      age: '',
    };
    console.disableYellowBox = true;
  }

  async componentDidMount() {
    AgeModule.loadModel();
  }

  
  getPictures() {
    this.payload_distance = "1 meter";
    console.log("resuming camera preview");
    this.camInterval = setInterval(this.takePicture, 1000);
  }

  stopPictures() {
    clearInterval(this.camInterval);
  }

  takePicture = async () => {
    if (this.camera) {
      const options = { quality: 0.5, base64: true, orientation: RNCamera.Constants.Orientation.portrait };
      const data = await this.camera.takePictureAsync(options);
      this.performAction(data.base64);
      // ImageResizer.createResizedImage(imageuri, 600, 600, 'JPEG', 2000)
      //   .then(response => {
      //     ImgToBase64.getBase64String(response.uri).then(base64String => {
      //       this.performAction(base64String);
      //     }).catch(errRead => {
      //       console.log(errRead);
      //     })
      //   })
      //   .catch(err => {
      //     console.log(err);
      //   });
    }
  };

  async performAction(imageData) {
    this.getNativeAge(imageData).then(age => {
      console.log(age);
      this.setState({age: age});
    }).catch(err => {
      console.log(err);
    })
  }

  getNativeAge(image) {
    return new Promise((resolve, reject) => {
      AgeModule.detectAge(image, function(err, text){
        if(err) {
          console.log('native age error ',err);
          reject(err);
          return;
        }
        console.log('native age detected', text);
        resolve(text);
      });
    });
  }
  
  render(){
    return (
      <>
        <StatusBar barStyle="dark-content" />
        <SafeAreaView>
          <ScrollView
            contentInsetAdjustmentBehavior="automatic"
            style={styles.scrollView}>
            <View style={styles.imageContainer}>
              <RNCamera
                style={[styles.preview, {
                  height: 100,
                  width: 100
                }]}
                ref={ref => { this.camera = ref; }}
                type={RNCamera.Constants.Type.back}
                flashMode={RNCamera.Constants.FlashMode.off}
                cameraViewDimensions={{ width: 100, height: 100 }}
                ratio="1:1"
                // onPictureTaken={d => {this.onPictureTakenEv(d)}}
                androidCameraPermissionOptions={{
                  title: 'Permission to use camera',
                  message: 'We need your permission to use your camera',
                  buttonPositive: 'Ok',
                  buttonNegative: 'Cancel',
                }}
                androidRecordAudioPermissionOptions={{
                  title: 'Permission to use audio recording',
                  message: 'We need your permission to use your audio',
                  buttonPositive: 'Ok',
                  buttonNegative: 'Cancel',
                }}
                playSoundOnCapture={false}
              >
              </RNCamera>
            </View>
            <View style={styles.body}>
            <View style={{ margin: 5 , width: '95%'}}>
                <Button id="idmodebtn" title="Start Feed" onPress={() => this.getPictures()} />
              </View>
              <View style={{ margin: 5 , width: '95%'}}>
                <Button title="Stop Feed" onPress={() => this.stopPictures()} />
              </View>
              <View>
                <Text>Detected Age: {this.state.age ? this.state.age : 'Unable to detect age.'}</Text>
              </View>
            </View>
          </ScrollView>
        </SafeAreaView>
      </>
    );
  }
};

const styles = StyleSheet.create({
  scrollView: {
    backgroundColor: Colors.lighter,
  },
  engine: {
    position: 'absolute',
    right: 0,
  },
  body: {
    backgroundColor: Colors.white,
  },
  sectionContainer: {
    marginTop: 32,
    paddingHorizontal: 24,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: '600',
    color: Colors.black,
  },
  sectionDescription: {
    marginTop: 8,
    fontSize: 18,
    fontWeight: '400',
    color: Colors.dark,
  },
  highlight: {
    fontWeight: '700',
  },
  footer: {
    color: Colors.dark,
    fontSize: 12,
    fontWeight: '600',
    padding: 4,
    paddingRight: 12,
    textAlign: 'right',
  },
  preview: {
    flex: 1,
    justifyContent: 'flex-end',
    alignItems: 'center',
  },
  imageContainer: {
    borderColor: 'blue',
    borderRadius: 0,
    alignItems: "center",
    width: viewPort.width,
    height: viewPort.width
  },
});

export default App;
