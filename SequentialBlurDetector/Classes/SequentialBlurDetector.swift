//
//  SequentialBlurDetector.swift
//  SequentialBlurDetector
//
//  Created by Petr Bobák on 25/02/2020.
//

import Foundation
import MLPatchExtractor
import UIKit

public class ImageBlurrinessResult {
    public var score: Float
    public var image: UIImage
    
    init(score: Float, image: UIImage) {
        self.score = score
        self.image = image
    }
}

public enum BlurrinessMeasure {
    case laplacianStandardDeviation
    case mlBlurrinessProbability
}

public class SequentialBlurDetector {
    public private(set) var measure: BlurrinessMeasure
    private var keepTopN: Int
    
    /// Property stores the N top results.
    public var topResults = [ImageBlurrinessResult]()
    
    /**
     Initialize `SequentialBlurDetector` with selected measure of blurriness.
     
     - Parameters:
        - measure: The specification of measure (one of `laplacianStandardDeviation` and `mlBlurrinessProbability`).
        - keepTopN: How many of the least blurred images to keep.
     */
    public init(measure: BlurrinessMeasure, keepTopN: Int = 2) {
        self.measure = measure
        self.keepTopN = keepTopN
    }
    
    /**
     Predict blurriness probability of `image` cropped by given mask and keep track of the least blurred images.
     
     - Parameters:
        - image: The source image to be evaluated.
        - patches: (Only for `mlBlurrinessProbability`) The number of patches to generate (in case of `uniform` the number of generated patches can be slightly larger or smaller in favor of uniform coverage).
        - sampling: (Only for `mlBlurrinessProbability`) The method to sample patches (`random` or `uniform`, for more details see `MLPatchSampling`).
        - maskRectangle: (Only for `mlBlurrinessProbability`) The image area to generate patches from.
        - completion: The completion block called after prediction is completed. Returns aggregated blurriness probability.
     */
    public func evaluate(image: UIImage, patches: Int, sampling: MLPatchSampling, maskRectangle: CGRect, completion: @escaping (SequentialBlurDetector, Float) -> Void) {
        switch measure {
        case .mlBlurrinessProbability:
            ImageBlurrinessMeasures.mlBlurrinessProbability(image: image, patches: patches, sampling: sampling, maskRectangle: maskRectangle) { (score) in
                self.processEvaluation(score: score, image: image, sense: <)
                completion(self, score)
            }
        case .laplacianStandardDeviation:
            ImageBlurrinessMeasures.laplacianStandardDeviation(image: image) { (score) in
                self.processEvaluation(score: score, image: image, sense: >)
                completion(self, score)
            }
        }
    }
    
    /// Clear the property of `topResults`.
    public func reset() {
        self.topResults = [ImageBlurrinessResult]()
    }
    
    /// Check if all topResults satisfy defined `predicate`.
    public func topResultsSatisfy(_ predicate: (ImageBlurrinessResult) -> (Bool)) -> Bool {
        guard topResults.count == keepTopN else {
            return false
        }
        
        return topResults.allSatisfy { predicate($0) }
    }
    
    private func processEvaluation(score: Float, image: UIImage, sense: ((Float, Float) -> Bool)) {
        let newResult = ImageBlurrinessResult(score: score, image: image)
        
        if let index = topResults.firstIndex(where: { sense(newResult.score, $0.score) }), index < keepTopN {
            topResults.insert(newResult, at: index)
        } else if topResults.isEmpty || topResults.count < keepTopN {
            topResults.append(newResult)
        }
        
        if topResults.count > keepTopN {
            topResults.removeLast()
        }
    }
}
