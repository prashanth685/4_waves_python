from pymongo import MongoClient
import datetime
from bson.objectid import ObjectId
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Database:
    def __init__(self, connection_string="mongodb://localhost:27017/", email="user@example.com"):
        try:
            self.client = MongoClient(connection_string)
            self.client.server_info()  # Test connection
            self.db = self.client["sarayu_db"]
            self.email = email
            self.email_safe = email.replace('@', '_').replace('.', '_')
            self.user_collection = self.db[f"user_{self.email_safe}"]
            self.tags_collection = self.db[f"tagcreated_{self.email_safe}"]
            self.messages_collection = self.db[f"mqttmessage_{self.email_safe}"]
            self.projects = []
            logging.info(f"Database initialized for {email}")
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {str(e)}")
            raise

    def close_connection(self):
        if self.client:
            self.client.close()
            logging.info("MongoDB connection closed.")

    def load_projects(self):
        self.projects.clear()
        try:
            for project in self.user_collection.find():
                project_name = project["project_name"]
                if project_name not in self.projects:
                    self.projects.append(project_name)
            logging.info(f"Loaded projects: {self.projects}")
            return self.projects
        except Exception as e:
            logging.error(f"Error loading projects: {str(e)}")
            return []

    def create_project(self, project_name):
        if not project_name:
            return False, "Project name cannot be empty!"
        if self.user_collection.find_one({"project_name": project_name}):
            return False, "Project already exists!"
        
        project_data = {
            "project_name": project_name,
            "created_at": datetime.datetime.now().isoformat()
        }
        try:
            self.user_collection.insert_one(project_data)
            if project_name not in self.projects:
                self.projects.append(project_name)
            logging.info(f"Project {project_name} created")
            return True, f"Project {project_name} created successfully!"
        except Exception as e:
            logging.error(f"Failed to create project: {str(e)}")
            return False, f"Failed to create project: {str(e)}"

    def edit_project(self, old_project_name, new_project_name):
        if new_project_name == old_project_name:
            return True, "No change made."
        if self.user_collection.find_one({"project_name": new_project_name}):
            return False, "Project already exists!"
        
        try:
            self.user_collection.update_one(
                {"project_name": old_project_name},
                {"$set": {"project_name": new_project_name}}
            )
            self.projects[self.projects.index(old_project_name)] = new_project_name
            self.tags_collection.update_many(
                {"project_name": old_project_name},
                {"$set": {"project_name": new_project_name}}
            )
            self.messages_collection.update_many(
                {"project_name": old_project_name},
                {"$set": {"project_name": new_project_name}}
            )
            logging.info(f"Project renamed from {old_project_name} to {new_project_name}")
            return True, f"Project renamed to {new_project_name} successfully!"
        except Exception as e:
            logging.error(f"Failed to edit project: {str(e)}")
            return False, f"Failed to edit project: {str(e)}"

    def delete_project(self, project_name):
        try:
            self.user_collection.delete_one({"project_name": project_name})
            self.tags_collection.delete_many({"project_name": project_name})
            self.messages_collection.delete_many({"project_name": project_name})
            if project_name in self.projects:
                self.projects.remove(project_name)
            logging.info(f"Project {project_name} deleted")
            return True, f"Project {project_name} deleted successfully!"
        except Exception as e:
            logging.error(f"Failed to delete project: {str(e)}")
            return False, f"Failed to delete project: {str(e)}"

    def get_project_data(self, project_name):
        try:
            data = self.user_collection.find_one({"project_name": project_name})
            logging.debug(f"Project data for {project_name}: {data}")
            return data
        except Exception as e:
            logging.error(f"Error fetching project data: {str(e)}")
            return None

    def parse_tag_string(self, tag_string):
        if not tag_string:
            logging.error("Tag cannot be empty!")
            return None
        return {"tag_name": tag_string}

    def add_tag(self, project_name, tag_data):
        if not self.get_project_data(project_name):
            return False, "Project not found!"
        if self.tags_collection.find_one({"project_name": project_name, "tag_name": tag_data["tag_name"]}):
            return False, "Tag already exists in this project!"
        
        tag_data["project_name"] = project_name
        tag_data["created_at"] = datetime.datetime.now().isoformat()
        try:
            self.tags_collection.insert_one(tag_data)
            logging.info(f"Tag {tag_data['tag_name']} added to {project_name}")
            return True, "Tag added successfully!"
        except Exception as e:
            logging.error(f"Failed to add tag: {str(e)}")
            return False, f"Failed to add tag: {str(e)}"

    def edit_tag(self, project_name, row, new_tag_data):
        tags = list(self.tags_collection.find({"project_name": project_name}))
        if row >= len(tags):
            return False, "Invalid tag index!"
        
        tag_id = tags[row]["_id"]
        current_tag_name = tags[row]["tag_name"]
        if new_tag_data["tag_name"] != current_tag_name and self.tags_collection.find_one(
            {"project_name": project_name, "tag_name": new_tag_data["tag_name"]}
        ):
            return False, "Tag already exists in this project!"
        
        new_tag_data["project_name"] = project_name
        new_tag_data["updated_at"] = datetime.datetime.now().isoformat()
        try:
            self.tags_collection.update_one(
                {"_id": tag_id},
                {"$set": new_tag_data}
            )
            self.messages_collection.update_many(
                {"project_name": project_name, "tag_name": current_tag_name},
                {"$set": {"tag_name": new_tag_data["tag_name"]}}
            )
            logging.info(f"Tag {current_tag_name} updated to {new_tag_data['tag_name']}")
            return True, "Tag updated successfully!"
        except Exception as e:
            logging.error(f"Failed to edit tag: {str(e)}")
            return False, f"Failed to edit tag: {str(e)}"

    def delete_tag(self, project_name, row):
        tags = list(self.tags_collection.find({"project_name": project_name}))
        if row >= len(tags):
            return False, "Invalid tag index!"
        
        tag_id = tags[row]["_id"]
        tag_name = tags[row]["tag_name"]
        try:
            self.tags_collection.delete_one({"_id": tag_id})
            self.messages_collection.delete_many({"project_name": project_name, "tag_name": tag_name})
            logging.info(f"Tag {tag_name} deleted from {project_name}")
            return True, "Tag deleted successfully!"
        except Exception as e:
            logging.error(f"Failed to delete tag: {str(e)}")
            return False, f"Failed to delete tag: {str(e)}"

    def update_tag_value(self, project_name, tag_name, values, timestamp=None):
        if not self.get_project_data(project_name):
            logging.error(f"Project {project_name} not found!")
            return False, "Project not found!"
        
        tag = self.tags_collection.find_one({"project_name": project_name, "tag_name": tag_name})
        if not tag:
            logging.error(f"Tag {tag_name} not found for project {project_name}!")
            return False, "Tag not found!"
        
        timestamp_str = timestamp if timestamp else datetime.datetime.now().isoformat()
        message_data = {
            "_id": ObjectId(),
            "topic": tag_name,
            "values": values,
            "project_name": project_name,
            "tag_name": tag_name,
            "timestamp": timestamp_str
        }
        try:
            result = self.messages_collection.insert_one(message_data)
            logging.debug(f"Inserted message with {len(values)} values for {tag_name} at {timestamp_str}: {result.inserted_id}")
            return True, "Tag values updated successfully!"
        except Exception as e:
            logging.error(f"Error inserting into mqttmessage_{self.email_safe}: {str(e)}")
            return False, f"Failed to update tag values: {str(e)}"

    def get_tag_values(self, project_name, tag_name):
        try:
            messages = list(self.messages_collection.find(
                {"project_name": project_name, "tag_name": tag_name}
            ).sort("timestamp", 1))
            if not messages:
                logging.debug(f"No messages found for {tag_name} in {project_name}")
                return []
            
            for msg in messages:
                if "timestamp" not in msg or "values" not in msg:
                    logging.warning(f"Invalid message format for {tag_name}: {msg}")
                    msg["timestamp"] = msg.get("timestamp", datetime.datetime.now().isoformat())
                    msg["values"] = msg.get("values", [])
            
            logging.debug(f"Retrieved {len(messages)} messages for {tag_name} in {project_name}")
            return messages
        except Exception as e:
            logging.error(f"Error fetching tag values for {tag_name} in {project_name}: {str(e)}")
            return []

    def save_tag_values(self, project_name, tag_name, data):
        if not self.get_project_data(project_name):
            logging.error(f"Project {project_name} not found!")
            return False, "Project not found!"
        
        tag = self.tags_collection.find_one({"project_name": project_name, "tag_name": tag_name})
        if not tag:
            logging.error(f"Tag {tag_name} not found for project {project_name}!")
            return False, "Tag not found!"
        
        message_data = {
            "_id": ObjectId(),
            "topic": tag_name,
            "values": data["values"],
            "project_name": project_name,
            "tag_name": tag_name,
            "timestamp": data["timestamp"]
        }
        try:
            result = self.messages_collection.insert_one(message_data)
            logging.debug(f"Saved {len(data['values'])} values for {tag_name} at {data['timestamp']}: {result.inserted_id}")
            return True, "Tag values saved successfully!"
        except Exception as e:
            logging.error(f"Error saving tag values for {tag_name}: {str(e)}")
            return False, f"Failed to save tag values: {str(e)}"