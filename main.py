from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware

# RecommendModel
class RecommendationSystem:
	def __init__(self, file_path):
		self.file_path = file_path
		self.df_rooms = self.load_rooms()
		self.vectorizer = TfidfVectorizer()
		self.tfidf_matrix = None
		self.update_tfidf()

	def load_rooms(self):
		try:
			df = pd.read_excel(self.file_path, sheet_name="Sheet1")
		except FileNotFoundError:
			df = pd.DataFrame(columns=["ID", "Name","Creator ID","Description", "Skills", "Position", "Tracks", "Combined_Features","CreatedAt","Members","CoverPicture"])  # لو الملف مش موجود، نخلق DataFrame فارغ
		df["Combined_Features"] = df[["Skills", "Position", "Tracks"]].fillna('').agg(' '.join, axis=1)
		return df

	def save_rooms(self):
		self.df_rooms.to_excel(self.file_path, index=False, sheet_name="Sheet1")
	
	def update_tfidf(self):
		self.tfidf_matrix = self.vectorizer.fit_transform(self.df_rooms["Combined_Features"])

	def add_new_room(self, room_id, name, skills, position, tracks,createdAt,members,coverPicture,desc):
		new_room = pd.DataFrame({
			"ID": [room_id], "Name": [name], "Description" : [desc],"Skills": [skills], "Position": [position], "Tracks": [tracks],"CreatedAt" : [createdAt],"Members" : [members], "CoverPicture" : [coverPicture]
		})
		new_room["Combined_Features"] = new_room[["Skills", "Position", "Tracks"]].fillna('').agg(' '.join, axis=1)
		
		self.df_rooms = pd.concat([self.df_rooms, new_room], ignore_index=True)
		self.update_tfidf()
		self.save_rooms() 

	def recommend_rooms(self, user_skills, user_position, top_n=4):
		user_profile = f"{user_skills} {user_position}"
		user_vector = self.vectorizer.transform([user_profile])
		similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
		recommended_indices = similarities.argsort()[-top_n:][::-1]
		return self.df_rooms.iloc[recommended_indices][["ID", "Name", "Skills", "Position", "Tracks","Description","CreatedAt","Members","CoverPicture"]]


file_path = r"./RoomData.xlsx"
rec_system = RecommendationSystem(file_path)


#Apis
app = FastAPI()
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"], 
	allow_credentials=True,
	allow_methods=["*"],  
	allow_headers=["*"],  
)

class UserInput(BaseModel):
	skills: str
	position: str
	
class NewRoom(BaseModel):
	coverPicture : str
	createdAt : str
	members : int	
	room_id: str
	name: str
	skills: str
	position: str
	tracks: str
	desc : str
	
@app.post("/recommend/")
def get_recommendations(user_input: UserInput):
	recommended_rooms = rec_system.recommend_rooms(user_input.skills, user_input.position)
	rooms_list = recommended_rooms.to_dict(orient='records')
	return {"recommended_rooms": rooms_list}
	
@app.post("/add_room/")
def add_room(new_room: NewRoom):
	rec_system.add_new_room(new_room.room_id, new_room.name, new_room.skills, new_room.position, new_room.tracks,new_room.createdAt,new_room.members,new_room.coverPicture,new_room.desc)
	return {"message": f"Room '{new_room.name}' added successfully!"}
	
@app.delete("/delete_room/{room_id}")
def delete_room(room_id: str):
	room_to_delete = rec_system.df_rooms[rec_system.df_rooms["ID"] == room_id]
	
	if room_to_delete.empty:
		return {"message": f"Room with ID {room_id} not found."}
	
	rec_system.df_rooms = rec_system.df_rooms[rec_system.df_rooms["ID"] != room_id]
	rec_system.update_tfidf() 
	return {"message": f"Room with ID {room_id} deleted successfully!"}
