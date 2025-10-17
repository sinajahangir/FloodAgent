
'''
This module defines a collection of classes representing various LLM-powered agents
designed to process and respond to user queries.

Only version implemented for now is CoordinateFloodProximityAgent
To be completed...

'''
import numpy as np
from scipy.spatial import cKDTree
import google.generativeai as genai
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from rasterio.warp import transform
from IPython.display import IFrame, display
import re
class CoordinateFloodProximityAgent:
    """
    An agent that uses an LLM to extract latitude and longitude from a user prompt
    and finds the coordinates of the closest flood pixel to that location.
    Social vulnerabity data is also extracted from a csv file.
    """

    def __init__(self,
                 flood_pixel_coords: np.ndarray, sovi_coords: np.ndarray, sovi_array:np.ndarray,\
                 pop_density_array:np.ndarray, BBN,llm_model,show_map=False):
        """
        Initializes the agent with building and flood location data.

        Args:
            flood_pixel_coords (np.ndarray): A NumPy array of shape (N, 2) where N
                                             is the number of flood pixels, and each
                                             row is [x, y] or [lon, lat] coordinates.
            sovi_coords (np.ndarray): A NumPy array of shape (M, 2) where M
                                      is the number of SoVI points, and each
                                      row is [x, y] or [lon, lat] coordinates.
            sovi_array (np.ndarray): A NumPy array of shape (M,) where M is the number of SoVI points.
            pop_density_array (np.ndarray): A NumPy array of shape (M,) where M is the number of population density points.
            BBN: A Bayesian belief network model. Used to infer infrastructure (road) flood vulnerability.
        """
        if llm_model is None:
            raise ValueError("AI agent is not initialized. Cannot create agent.")

        self.llm = llm_model

        self.BBN = BBN
        self.infer = VariableElimination(self.BBN)
        self.map=show_map


        # Store flood pixel coordinates
        self.flood_pixels = flood_pixel_coords
        self.sovi_pixels = sovi_coords
        self.sovi_array = sovi_array
        self.pop_density_array = pop_density_array
        print(f"Stored {len(self.flood_pixels)} flood pixel coordinates.")

        # Build a k-d tree for *flood pixels* for efficient nearest neighbor search
        if len(self.flood_pixels) > 0 and len(self.sovi_pixels)>0:
             print("Building k-d tree for flood pixels...")
             # Assuming flood_pixel_coords are [Lon, Lat] or [X, Y]
             self.flood_kdtree = cKDTree(self.flood_pixels)
             self.sovi_kdtree = cKDTree(self.sovi_pixels)
             print("Flood k-d tree built.")
        else:
             print("Warning: No flood pixels or SoVI was provided. Closest pixel search will not work.")
             self.flood_kdtree = None
             self.sovi_kdtree = None
    def _infer_infrastructure_flood_vulnerability(self, rp='Moderate'):
        # To be completed
        return None
    def generate_google_map_link(self,latitude_str: str, longitude_str: str, zoom_level: int = 14) -> dict:
        """
        Takes latitude and longitude as strings, validates them, and generates 
        Google Maps URLs.

        Args:
            latitude_str: The latitude value as a string (e.g., "34.0522").
            longitude_str: The longitude value as a string (e.g., "-118.2437").
            zoom_level: The desired zoom level (1-20). Default is 14.

        Returns:
            A dictionary containing the generated URLs or an error message.
            Example success:
            {
                "success": True,
                "direct_link": "...",
                "embed_link": "..."
            }
            Example failure:
            {
                "success": False,
                "error": "..."
            }
        """
        try:
            # 1. Convert to float
            lat = float(latitude_str.strip())
            lon = float(longitude_str.strip())
        except ValueError:
            return {
                "success": False,
                "error": "Invalid input format. Latitude and Longitude must be numeric strings."
            }

        # 2. Validate Coordinate Ranges
        if not (-90 <= lat <= 90):
            return {
                "success": False,
                "error": f"Invalid Latitude: {lat}. Must be between -90 and 90."
            }
        if not (-180 <= lon <= 180):
            return {
                "success": False,
                "error": f"Invalid Longitude: {lon}. Must be between -180 and 180."
            }
    
        # 3. Generate URLs
        # Direct View Link: uses the /@lat,lon,zoomz format
        direct_link = f"https://www.google.com/maps/@{lat},{lon},{zoom_level}z"
        
        # Embed Link (for iframes): uses the ?q=lat,lon&output=embed format
        embed_link = f"https://maps.google.com/maps?q={lat},{lon}&output=embed"
        
        # 4. Return results
        return {
            "success": True,
            "direct_link": direct_link,
            "embed_link": embed_link
        }



    def _extract_lat_lon_from_prompt(self, prompt: str) -> tuple[float, float] | None:
        """Uses the LLM to extract latitude and longitude from the user prompt."""
        # (This function is identical to the one in the previous LocationMappingAgent)

        llm_prompt = f"""
        Analyze the following user query and extract the single pair of geographical coordinates (latitude and longitude) mentioned.
        Latitude must be between 50 and 52. Longitude must be between -113 and -115.
        Pay attention to signs (N/S, E/W) or negative values indicating direction.

        User query: "{prompt}"

        Respond ONLY with the extracted coordinates in the format:
        LATITUDE=value, LONGITUDE=value
        For example: LATITUDE=40.7128, LONGITUDE=-74.0060

        If you cannot reliably extract both a valid latitude and a valid longitude from the query, and the question is not related to flood at that location, respond with the exact phrase "Not in the database".
        """
        #print(f"Sending prompt to LLM for coordinate extraction:\n---\n{llm_prompt}\n---")

        try:
            response = self.llm.generate_content(llm_prompt)
            extracted_text = response.text.strip()
            #print(f"LLM response for coordinate extraction: '{extracted_text}'")

            if extracted_text == "Not in the database":
                #print("LLM indicated coordinates could not be found.")
                return None

            match = re.match(r"LATITUDE=(-?[\d.]+),\s*LONGITUDE=(-?[\d.]+)", extracted_text, re.IGNORECASE)

            if match:
                lat_str, lon_str = match.groups()
                if self.map:
                    map_data = self.generate_google_map_link(
                        latitude_str=lat_str, 
                        longitude_str=lon_str, 
                        zoom_level=10 # Set a moderate zoom level
                    )
                    if map_data["success"]:
                        embed_url = map_data["embed_link"]

                        print(f"Displaying Location: Lat={lat_str}, Lon={lon_str}")
                        print(f"Direct Link (Open in New Tab): {map_data['direct_link']}")
                        
                        # Create the IFrame object
                        # The IFrame constructor takes the URL, width, and height.
                        map_iframe = IFrame(src=embed_url, width='100%', height=450)
                        
                        # Render the IFrame in the cell output
                        display(map_iframe)
                try:
                    latitude = float(lat_str)
                    longitude = float(lon_str)
                    if -90 <= latitude <= 90 and -180 <= longitude <= 180:
                        # Determine UTM EPSG based on lon/lat
                        utm_zone = int((longitude + 180) / 6) + 1
                        is_northern = latitude >= 0
                        utm_epsg = 32600 + utm_zone if is_northern else 32700 + utm_zone
                        utm_crs = f"EPSG:{utm_epsg}"

                        # Convert to UTM using rasterio
                        utm_x, utm_y = transform("EPSG:4326", utm_crs, [longitude], [latitude])
                        print(f"Converted to UTM ({utm_crs}): Easting={utm_x[0]}, Northing={utm_y[0]}")

                        return utm_x[0], utm_y[0]
                    else:
                        print(f"Extracted coordinates out of valid range: Lat={latitude}, Lon={longitude}")
                        return None
                except ValueError:
                    print(f"Could not convert extracted strings to float: '{lat_str}', '{lon_str}'")
                    return None
            else:
                print(f"LLM response did not match expected format 'LATITUDE=..., LONGITUDE=...': '{extracted_text}'")
                return None

        except Exception as e:
            #print(f"Error during LLM call or parsing for coordinate extraction: {e}")
            return None


    def _find_closest_flood_pixel(self, target_lat: float, target_lon: float) -> tuple[np.ndarray | None, float | None]:
        """
        Finds the nearest flood pixel using the flood k-d tree.

        Args:
            target_lat: The target latitude.
            target_lon: The target longitude.

        Returns:
            A tuple (closest_pixel_coords, distance) or (None, None) if error/no data.
            closest_pixel_coords is [longitude, latitude] or [x, y] as in the input array.
        """
        if self.flood_kdtree is None or len(self.flood_pixels) == 0:
            print("No flood pixel data or k-d tree available for search.")
            return None, None
        if self.sovi_kdtree is None or len(self.sovi_pixels) == 0:
            print("No SoVI data or k-d tree available for search.")
            return None, None

        # Ensure target coords are in the correct order for the tree ([Lon, Lat] or [X,Y])
        target_point = np.array([target_lon, target_lat])

        try:
            # Query the k-d tree: find the 1 nearest neighbor
            distance, index = self.flood_kdtree.query(target_point, k=1)
            distance_sovi, index_sovi = self.sovi_kdtree.query(target_point, k=1)
            sovi_value = self.sovi_array[index_sovi]
            density_value = self.pop_density_array[index_sovi]
            closest_pixel_coords = self.flood_pixels[index]
            print(f"Closest flood pixel found: Coords={closest_pixel_coords}, Distance={distance:.4f}")
            return closest_pixel_coords, distance, sovi_value, density_value
        except Exception as e:
            print(f"Error during flood k-d tree query: {e}")
            return None, None


    def find_closest_flood_pixel_to_location(self, user_prompt: str) -> str:
        """
        Processes a user prompt to extract coordinates and find the nearest flood pixel.

        Args:
            user_prompt (str): The natural language query from the user containing coordinates.

        Returns:
            str: A natural language response summarizing the findings.
        """
        # 1. Extract Lat/Lon using LLM
        extracted_coords = self._extract_lat_lon_from_prompt(user_prompt)

        if extracted_coords is None:
            # Ask LLM to formulate a response about not finding coordinates
            try:
                 response = self.llm.generate_content(f"The user asked: '{user_prompt}'. I could not extract valid flood latitude and longitude coordinates from this query. Please formulate a polite response asking the user to provide clear coordinates (e.g., 'latitude 40.7, longitude -74.0') for flooding.")
                 return response.text.strip()
            except Exception as e:
                 print(f"LLM failed to generate clarification response: {e}")
                 return "I couldn't identify valid geographic coordinates (latitude and longitude) in your request. Please provide them clearly."

        target_lon, target_lat = extracted_coords

        # 2. Find the closest FLOOD PIXEL computationally
        closest_pixel_coords, distance, sovi_value, density_value = self._find_closest_flood_pixel(target_lat, target_lon)
        rc = self.infer.query(variables=['rc'],evidence={'pop_d_c':density_value})
        rc_class=rc.state_names['rc'][np.where(rc.values==max(rc.values))[0][0]] # the road class associated with the highest probability

        fec = self.infer.query(variables=['f_exp_c'],evidence={'pop_d_c':density_value,'rc':rc_class})
        fec_class=fec.state_names['f_exp_c'][np.where(fec.values==max(fec.values))[0][0]] # the fload exposure class associated with the highest probability




        # 3. Formulate the final response (optionally using LLM)
        if closest_pixel_coords is not None and distance is not None:
            # Assuming coords are [Lon, Lat] for reporting
            summary = (f"For the location you provided (approx. Latitude={target_lat:.6f}, Longitude={target_lon:.6f}), "
                       f"the closest flood pixel recorded in the data is at Latitude={closest_pixel_coords[1]:.6f}, Longitude={closest_pixel_coords[0]:.6f}. "
                       f"The calculated distance is about {distance:.4f} units (based on the coordinate system).")

            # Optional: Use LLM to make the response more conversational
            try:
                response_prompt = f"""
                I searched the flood data and found the closest recorded flood pixel is at coordinates Latitude={closest_pixel_coords[1]:.6f}, Longitude={closest_pixel_coords[0]:.6f}.
                The distance between the user's point and this flood pixel is {distance:.4f} meters (UTM). The SoVI value class is {sovi_value:s} based on the NRCan social fabric product.
                The population density is {density_value:s} people per square kilometer. The expected road class for the associated flood pixel is {rc_class:s}. Based on the population density, and road class, the flood exposure is {fec_class:s}.

                Generate a concise response for the user, incorporating this information. Based on the SoVI value class, change your tone of response. Based on the distance, the SoVI value, expected road class, population density, and flood exposure class, estimate flood resilience.
                Explain your chain of thoughts for how you decided what the flood resilience is for that location.
                """
                final_response = self.llm.generate_content(response_prompt)
                return final_response.text.strip()
            except Exception as e:
                print(f"LLM failed to generate final response: {e}")
                # Fallback to the pre-formatted summary
                return summary

        elif self.flood_kdtree is None:
             # Case where no flood data was loaded
             return f"I understood the location (Lat={target_lat:.6f}, Lon={target_lon:.6f}), but I don't have any flood pixel data loaded to search against."
        else:
            # Case where search failed for other reasons
             return f"I extracted the coordinates (Lat={target_lat:.6f}, Lon={target_lon:.6f}), but encountered an error trying to find the closest flood pixel in the data."