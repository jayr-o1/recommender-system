import { db } from "../firebase";
import {
    collection,
    doc,
    getDoc,
    updateDoc,
    serverTimestamp,
    setDoc,
    query,
    where,
    getDocs,
    deleteDoc,
    writeBatch,
} from "firebase/firestore";

// API Configuration
const API_CONFIG = {
    endpoints: {
        recommend: "http://localhost:8000/recommend", // Main recommendation endpoint
        simpleRecommend: "http://localhost:8000/api/recommend", // Frontend-friendly endpoint
        health: "http://localhost:8000/health", // Health check endpoint
        fields: "http://localhost:8000/fields", // Get available fields
        specializations: "http://localhost:8000/specializations", // Get available specializations
        matchSkill: "http://localhost:8000/match_skill", // Skill matching endpoint
    },
    timeout: 30000, // 30 second timeout
};

/**
 * Helper function to save recommendation data to a JSON file in the current directory
 * @param {Object} data - The recommendation data to save
 * @param {string} prefix - Prefix for the filename
 */
const saveRecommendationToFile = (data, prefix = "recommendation") => {
    try {
        // Load fs module
        const fs = require("fs");
        const path = require("path");

        // Create filename with timestamp
        const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
        const filename = `${prefix}_${timestamp}.json`;

        // Convert data to formatted JSON string
        const jsonData = JSON.stringify(data, null, 2);

        // Write to file in current directory
        fs.writeFileSync(filename, jsonData);

        console.log(
            `Saved recommendation data to ${filename} in current directory`
        );
        return true;
    } catch (error) {
        console.error("Error saving recommendation to file:", error);
        return false;
    }
};

/**
 * Helper function to fetch with timeout
 */
const fetchWithTimeout = async (url, options, timeout) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal,
        });
        clearTimeout(timeoutId);
        return response;
    } catch (error) {
        clearTimeout(timeoutId);
        if (error.name === "AbortError") {
            throw new Error(`Request timed out after ${timeout}ms`);
        }
        throw error;
    }
};

/**
 * Check API health
 * @returns {Promise} - Promise that resolves to health status
 */
export const checkApiHealth = async () => {
    try {
        const response = await fetchWithTimeout(
            API_CONFIG.endpoints.health,
            {
                method: "GET",
                headers: {
                    "Content-Type": "application/json",
                },
            },
            API_CONFIG.timeout
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                `API health check failed with status ${response.status}`
            );
        }

        return {
            success: true,
            status: data,
        };
    } catch (error) {
        console.error("Error checking API health:", error);
        return {
            success: false,
            message: error.message || "Failed to check API health",
            status: null,
        };
    }
};

/**
 * Get all available career fields
 * @returns {Promise} - Promise that resolves to list of fields
 */
export const getCareerFields = async () => {
    try {
        const response = await fetchWithTimeout(
            API_CONFIG.endpoints.fields,
            {
                method: "GET",
                headers: {
                    "Content-Type": "application/json",
                },
            },
            API_CONFIG.timeout
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                `Failed to get fields with status ${response.status}`
            );
        }

        return {
            success: true,
            fields: data,
        };
    } catch (error) {
        console.error("Error getting career fields:", error);
        return {
            success: false,
            message: error.message || "Failed to get career fields",
            fields: null,
        };
    }
};

/**
 * Get all available specializations
 * @returns {Promise} - Promise that resolves to list of specializations
 */
export const getCareerSpecializations = async () => {
    try {
        const response = await fetchWithTimeout(
            API_CONFIG.endpoints.specializations,
            {
                method: "GET",
                headers: {
                    "Content-Type": "application/json",
                },
            },
            API_CONFIG.timeout
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                `Failed to get specializations with status ${response.status}`
            );
        }

        return {
            success: true,
            specializations: data,
        };
    } catch (error) {
        console.error("Error getting career specializations:", error);
        return {
            success: false,
            message: error.message || "Failed to get career specializations",
            specializations: null,
        };
    }
};

/**
 * Match a user skill against a standard skill
 * @param {string} userSkill - The user's skill name
 * @param {string} standardSkill - The standard skill to match against
 * @param {boolean} useSemantic - Whether to use semantic matching (default: true)
 * @returns {Promise} - Promise that resolves to match result
 */
export const matchSkill = async (
    userSkill,
    standardSkill,
    useSemantic = true
) => {
    try {
        const response = await fetchWithTimeout(
            API_CONFIG.endpoints.matchSkill,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    user_skill: userSkill,
                    standard_skill: standardSkill,
                    use_semantic: useSemantic,
                }),
            },
            API_CONFIG.timeout
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail ||
                    `Skill matching failed with status ${response.status}`
            );
        }

        return {
            success: true,
            result: data,
        };
    } catch (error) {
        console.error("Error matching skills:", error);
        return {
            success: false,
            message: error.message || "Failed to match skills",
            result: null,
        };
    }
};

/**
 * Get career recommendations with proficiency levels (simplified API for frontend)
 * @param {Array} skills - Array of skill objects with name and proficiency
 * @param {Object} options - Optional parameters for the recommendation
 * @returns {Promise} - Promise that resolves to recommendations
 */
export const getCareerRecommendations = async (skills, options = {}) => {
    try {
        // Validate input
        if (!skills || !Array.isArray(skills)) {
            throw new Error("Skills must be an array");
        }

        // Format skills as {skillName: proficiency} dictionary
        const skillsDict = {};
        skills.forEach((skill) => {
            if (!skill.name) {
                throw new Error("Each skill must have a name");
            }
            // Convert proficiency to number and ensure it's within 1-100 range
            const proficiency = Math.min(
                Math.max(parseInt(skill.proficiency) || 50, 1),
                100
            );
            skillsDict[skill.name] = proficiency;
        });

        console.log("Sending skills to frontend-friendly API:", skillsDict);

        // Call the API with timeout
        const response = await fetchWithTimeout(
            API_CONFIG.endpoints.simpleRecommend,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    skills: skillsDict,
                    currentField: options.currentField || null,
                    currentSpecialization:
                        options.currentSpecialization || null,
                }),
            },
            API_CONFIG.timeout
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail ||
                    `API request failed with status ${response.status}`
            );
        }

        // Save recommendation data to file if successful
        const result = {
            success: true,
            recommendations: data.recommendations || data,
            timestamp: new Date().toISOString(),
            inputSkills: skills.map((s) => ({
                name: s.name,
                proficiency: s.proficiency,
            })),
            options,
        };

        // Save to JSON file if saveToFile option is true
        if (options.saveToFile) {
            saveRecommendationToFile(result, "simple_recommendation");
        }

        return result;
    } catch (error) {
        console.error("Error getting recommendations:", error);
        return {
            success: false,
            message: error.message || "Failed to get recommendations",
            recommendations: null,
        };
    }
};

/**
 * Get detailed career recommendations with advanced options
 * @param {Array} skills - Array of skill objects with name and proficiency
 * @param {Object} options - Optional parameters for the recommendation
 * @param {boolean} options.saveToFile - Whether to save recommendations to a file
 * @param {number} options.topFields - Number of top fields to return
 * @param {number} options.topSpecializations - Number of top specializations to return
 * @param {number} options.fuzzyThreshold - Threshold for fuzzy matching
 * @param {boolean} options.simplifiedResponse - Whether to simplify the response
 * @param {boolean} options.useSemantic - Whether to use semantic matching
 * @returns {Promise} - Promise that resolves to detailed recommendations
 */
export const getDetailedCareerRecommendations = async (
    skills,
    options = {}
) => {
    try {
        // Validate input
        if (!skills || !Array.isArray(skills)) {
            throw new Error("Skills must be an array");
        }

        // Format skills as {skillName: proficiency} dictionary
        const skillsDict = {};
        skills.forEach((skill) => {
            if (!skill.name) {
                throw new Error("Each skill must have a name");
            }
            // Convert proficiency to number and ensure it's within 1-100 range
            const proficiency = Math.min(
                Math.max(parseInt(skill.proficiency) || 50, 1),
                100
            );
            skillsDict[skill.name] = proficiency;
        });

        console.log("Sending skills dictionary to detailed API:", skillsDict);

        // Prepare API request body with params directly in the root (not nested)
        const requestBody = {
            skills: skillsDict,
            // Add the params directly at the root level of the request
            top_fields: options.topFields || 3,
            top_specializations: options.topSpecializations || 3,
            fuzzy_threshold: options.fuzzyThreshold || 80,
            simplified_response: options.simplifiedResponse || false,
            use_semantic:
                options.useSemantic !== undefined ? options.useSemantic : true,
        };

        // Call the API with timeout
        const response = await fetchWithTimeout(
            API_CONFIG.endpoints.recommend,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(requestBody),
            },
            API_CONFIG.timeout
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail ||
                    `API request failed with status ${response.status}`
            );
        }

        return {
            success: true,
            recommendations: data,
            fileSaved: options.saveToFile || false,
        };
    } catch (error) {
        console.error("Error getting detailed recommendations:", error);
        return {
            success: false,
            message: error.message || "Failed to get detailed recommendations",
            recommendations: null,
            fileSaved: false,
        };
    }
};

/**
 * Save career recommendation to Firestore
 */
export const saveCareerRecommendation = async (
    userId,
    universityId,
    recommendationData
) => {
    try {
        // Validate input
        if (!userId || !universityId || !recommendationData) {
            throw new Error("Missing required parameters");
        }

        // Get employee document reference
        const { employeeRef, employeeDocId } = await getEmployeeReference(
            userId,
            universityId
        );

        // Create recommendation object
        const recommendationId = `rec_${Date.now()}`;
        const recommendationObj = {
            id: recommendationId,
            timestamp: serverTimestamp(),
            ...recommendationData,
            status: "active",
            employeeId: userId.startsWith("emp_")
                ? userId.split("_")[1]
                : userId,
            createdBy: userId,
            createdAt: serverTimestamp(),
        };

        // Create a batch write for atomic operation
        const batch = writeBatch(db);

        // Reference to recommendation document
        const recommendationRef = doc(
            db,
            "universities",
            universityId,
            "employees",
            employeeDocId,
            "careerRecommendations",
            recommendationId
        );

        // Add operations to batch
        batch.set(recommendationRef, recommendationObj);
        batch.update(employeeRef, {
            latestRecommendationId: recommendationId,
            lastRecommendationDate: serverTimestamp(),
            updatedAt: serverTimestamp(),
        });

        // Commit the batch
        await batch.commit();

        return {
            success: true,
            recommendationId,
        };
    } catch (error) {
        console.error("Error saving recommendation:", error);
        return {
            success: false,
            message: error.message,
        };
    }
};

/**
 * Helper function to get employee reference
 */
const getEmployeeReference = async (userId, universityId) => {
    let employeeDocId = userId;

    if (userId.startsWith("emp_")) {
        const employeeId = userId.split("_")[1];
        const employeesRef = collection(
            db,
            "universities",
            universityId,
            "employees"
        );
        const q = query(employeesRef, where("employeeId", "==", employeeId));
        const querySnapshot = await getDocs(q);

        if (querySnapshot.empty) {
            throw new Error("Employee record not found");
        }
        employeeDocId = querySnapshot.docs[0].id;
    }

    const employeeRef = doc(
        db,
        "universities",
        universityId,
        "employees",
        employeeDocId
    );

    return { employeeRef, employeeDocId };
};

/**
 * Get latest career recommendation
 */
export const getLatestCareerRecommendation = async (userId, universityId) => {
    try {
        const { employeeRef } = await getEmployeeReference(
            userId,
            universityId
        );
        const employeeDoc = await getDoc(employeeRef);

        if (!employeeDoc.exists()) {
            throw new Error("Employee not found");
        }

        const { latestRecommendationId } = employeeDoc.data();

        if (!latestRecommendationId) {
            return {
                success: false,
                message: "No recommendations found",
                recommendation: null,
            };
        }

        const recommendationRef = doc(
            db,
            "universities",
            universityId,
            "employees",
            employeeDoc.id,
            "careerRecommendations",
            latestRecommendationId
        );

        const recommendationDoc = await getDoc(recommendationRef);

        if (!recommendationDoc.exists()) {
            throw new Error("Recommendation not found");
        }

        return {
            success: true,
            recommendation: recommendationDoc.data(),
        };
    } catch (error) {
        console.error("Error getting recommendation:", error);
        return {
            success: false,
            message: error.message,
            recommendation: null,
        };
    }
};

/**
 * Delete latest career recommendation
 */
export const deleteLatestCareerRecommendation = async (
    userId,
    universityId
) => {
    try {
        const { employeeRef } = await getEmployeeReference(
            userId,
            universityId
        );
        const employeeDoc = await getDoc(employeeRef);

        if (!employeeDoc.exists()) {
            throw new Error("Employee not found");
        }

        const { latestRecommendationId } = employeeDoc.data();

        if (!latestRecommendationId) {
            return {
                success: false,
                message: "No recommendations to delete",
            };
        }

        // Create a batch write for atomic operation
        const batch = writeBatch(db);

        // Delete recommendation
        const recommendationRef = doc(
            db,
            "universities",
            universityId,
            "employees",
            employeeDoc.id,
            "careerRecommendations",
            latestRecommendationId
        );

        batch.delete(recommendationRef);
        batch.update(employeeRef, {
            latestRecommendationId: null,
            lastRecommendationDate: null,
            updatedAt: serverTimestamp(),
        });

        await batch.commit();

        return {
            success: true,
            message: "Recommendation deleted successfully",
        };
    } catch (error) {
        console.error("Error deleting recommendation:", error);
        return {
            success: false,
            message: error.message,
        };
    }
};
