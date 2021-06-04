//
//  ImageBlurMeasures.swift
//  SequentialBlurDetector
//
//  Created by Petr Bobák on 25/02/2020.
//

import Foundation
import Accelerate
@_exported import BlurDetector
import MLPatchExtractor
import UIKit

class ImageBlurrinessMeasures {
    static func mlBlurrinessProbability(image: UIImage, patches: Int, sampling: MLPatchSampling, maskRectangle: CGRect, completion: @escaping (Float) -> Void) {
        BlurDetector.evaluate(image: image, patches: patches,sampling: sampling, maskRectangle: maskRectangle) { blurrinessProbability, _  in
            completion(Float(blurrinessProbability))
        }
    }
    
    static func laplacianStandardDeviation(image: UIImage, completion: @escaping (Float) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            guard let cgImage = image.cgImage else {
                fatalError("Unable to get CGImage")
            }
            
            guard let _ = vImage_CGImageFormat(cgImage: cgImage) else {
                fatalError("Unable to get color space")
            }

            // Create source buffer
            guard var sourceBuffer = try? vImage_Buffer(cgImage: cgImage) else {
                fatalError("Unable to create sourceBuffer")
            }
            defer {
                sourceBuffer.free()
            }
            
            // Create source grayscale buffer
            guard var sourceGrayscaleBuffer = try? vImage_Buffer(width: Int(sourceBuffer.width), height: Int(sourceBuffer.height), bitsPerPixel: 8) else {
                fatalError("Unable to create destination buffers.")
            }
            defer {
                sourceGrayscaleBuffer.free()
            }
            
            // Convert to grayscale
            vImage_ARGB8888ToPlanar8Grayscale(&sourceBuffer, &sourceGrayscaleBuffer)

            // Create Floating Point Pixels to Use in vDSP
            // vImage buffers store their image data in row major format. However, when you are passing data between vImage and vDSP, be aware that, in some cases, vImage will add extra bytes at the end of each row to maximize performance.
            var floatPixels: [Float]
            let width = Int(sourceGrayscaleBuffer.width)
            let height = Int(sourceGrayscaleBuffer.height)
            let count = width * height

            if sourceGrayscaleBuffer.rowBytes == width * MemoryLayout<Pixel_8>.stride {
                // In some cases, this disparity between the row bytes used to hold image data and the buffer's actual row bytes may not affect your app's results. For this sample, compare the destination's rowBytes property against its width, multiplied by the stride of a Pixel_8 and, if the values are the same, you can infer there's no row byte padding and simply pass a pointer to the vImage buffer's data to vDSP's integerToFloatingPoint method
                let start = sourceGrayscaleBuffer.data.assumingMemoryBound(to: Pixel_8.self)
                floatPixels = vDSP.integerToFloatingPoint(
                    UnsafeMutableBufferPointer(
                        start: start,
                        count: count),
                        floatingPointType: Float.self)
            } else {
                //  In the case where there is row byte padding, create an intermediate vImage buffer with explicit row bytes and use the vImage vImageConvert_Planar8toPlanarF function to populate floatPixels:
                floatPixels = [Float](unsafeUninitializedCapacity: count) { buffer, initializedCount in
                    var floatBuffer = vImage_Buffer(
                                            data: buffer.baseAddress,
                                            height: sourceGrayscaleBuffer.height,
                                            width: sourceGrayscaleBuffer.width,
                                            rowBytes: width * MemoryLayout<Float>.size)
                    vImageConvert_Planar8toPlanarF(&sourceGrayscaleBuffer, &floatBuffer, 0, 255, vImage_Flags(kvImageNoFlags))
                    initializedCount = count
                }
            }

            let laplacian: [Float] =
                [-1, -1, -1,
                 -1,  8, -1,
                 -1, -1, -1]


            // Convolve with Laplacian
            vDSP.convolve(floatPixels, rowCount: height, columnCount: width, with3x3Kernel: laplacian, result: &floatPixels)

            var mean = Float.nan
            var stdDev = Float.nan
            
            // Calculate standard deviation
            vDSP_normalize(floatPixels, 1, nil, 1, &mean, &stdDev, vDSP_Length(count))

            // Return score
            completion(stdDev * stdDev)
        }
    }
    
    static private func vImage_ARGB8888ToPlanar8Grayscale(_ src: UnsafePointer<vImage_Buffer>, _ dest: UnsafePointer<vImage_Buffer>) {
        
        // Create a 1D matrix containing the three luma coefficients that specify the color-to-grayscale conversion.
        let redCoefficient: Float = 0.2126
        let greenCoefficient: Float = 0.7152
        let blueCoefficient: Float = 0.0722
        
        // The matrix multiply function requires an Int32 divisor, but the coefficients are Float values. To simplify the matrix initialization, declare and use fDivisor to multiply each coefficients by the divisor.
        let divisor: Int32 = 0x1000
        let fDivisor = Float(divisor)

        var coefficientsMatrix = [
            Int16(redCoefficient * fDivisor),
            Int16(greenCoefficient * fDivisor),
            Int16(blueCoefficient * fDivisor)
        ]
        
        // Use the matrix of coefficients to compute the scalar luminance by returning the dot product of each RGB pixel and the coefficients matrix.
        let preBias: [Int16] = [0, 0, 0, 0]
        let postBias: Int32 = 0

        vImageMatrixMultiply_ARGB8888ToPlanar8(
            src,
            dest,
            &coefficientsMatrix,
            divisor,
            preBias,
            postBias,
            vImage_Flags(kvImageNoFlags))
    }
}
